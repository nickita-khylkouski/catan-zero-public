use catanatron_rs::{
    ActionPrompt, ActionSpace, Color, Game, MapKind, NumberPlacement, Player, action_type_index,
    board_tensor_shape, create_board_tensor_flat, create_sample, feature_ordering,
    game_to_json_value, legal_action_indices_with_space, legal_action_mask_from_indices,
};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const TURNS_LIMIT: usize = 1000;
const DISCOUNT_FACTOR: f64 = 0.99;

#[derive(Debug)]
struct Config {
    num_games: usize,
    players: Vec<String>,
    tournament_players: Option<Vec<String>>,
    seed: Option<u64>,
    map_kind: MapKind,
    number_placement: NumberPlacement,
    discard_limit: u8,
    friendly_robber: bool,
    vps_to_win: i16,
    jsonl_path: Option<PathBuf>,
    csv_dir: Option<PathBuf>,
    include_board_tensor: bool,
    parallel: bool,
    quiet: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_games: 1,
            players: vec!["R".to_string(), "R".to_string()],
            tournament_players: None,
            seed: None,
            map_kind: MapKind::Base,
            number_placement: NumberPlacement::OfficialSpiral,
            discard_limit: 7,
            friendly_robber: false,
            vps_to_win: 10,
            jsonl_path: None,
            csv_dir: None,
            include_board_tensor: false,
            parallel: true,
            quiet: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SimulationStats {
    completed_turns: usize,
    wins: usize,
}

#[derive(Clone, Debug, Default)]
struct TournamentScore {
    games: usize,
    wins: usize,
    draws: usize,
    turns: usize,
}

#[derive(Clone, Debug, Default)]
struct TournamentStats {
    total_games: usize,
    draws: usize,
    completed_turns: usize,
    scores: Vec<TournamentScore>,
}

struct NativeRecord {
    game_id: usize,
    step_id: usize,
    state_index: usize,
    color: Color,
    current_prompt: ActionPrompt,
    sample: Vec<f64>,
    action_index: usize,
    action_type_index: usize,
    legal_action_indices: Vec<usize>,
    legal_action_mask: Vec<u8>,
    board_tensor: Option<Vec<f64>>,
}

struct NativeCsvWriter {
    samples: BufWriter<File>,
    actions: BufWriter<File>,
    rewards: BufWriter<File>,
    metadata: BufWriter<File>,
    legal_actions: BufWriter<File>,
    legal_action_masks: BufWriter<File>,
    main: BufWriter<File>,
    board_tensors: Option<BufWriter<File>>,
    feature_names: Vec<String>,
    action_space_size: usize,
    wrote_header: bool,
    include_board_tensor: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args(std::env::args().skip(1))?;
    if let Some(roster) = &config.tournament_players {
        return run_tournament(&config, roster);
    }

    let colors = [Color::Red, Color::Blue, Color::White, Color::Orange];
    let player_colors = &colors[..config.players.len()];
    let player_templates = build_player_templates(&config.players, player_colors)?;
    let use_parallel = config.parallel && config.jsonl_path.is_none() && config.csv_dir.is_none();
    let mut writer = match &config.jsonl_path {
        Some(path) => {
            let path = output_path(path)?;
            Some(BufWriter::new(
                File::create(path).map_err(|error| error.to_string())?,
            ))
        }
        None => None,
    };
    let mut csv_writer = match &config.csv_dir {
        Some(path) => Some(NativeCsvWriter::new(
            path,
            config.include_board_tensor,
            config.players.len(),
            config.map_kind,
            player_colors,
        )?),
        None => None,
    };

    let start = Instant::now();
    let stats = if use_parallel {
        play_parallel_stats(&config, &player_templates)?
    } else {
        play_serial(
            &config,
            player_colors,
            &player_templates,
            &mut writer,
            &mut csv_writer,
        )?
    };

    if let Some(writer) = writer.as_mut() {
        writer.flush().map_err(|error| error.to_string())?;
    }
    if let Some(csv_writer) = csv_writer.as_mut() {
        csv_writer.flush()?;
    }

    let elapsed = start.elapsed();
    let games_per_second = config.num_games as f64 / elapsed.as_secs_f64();
    let turns_per_second = stats.completed_turns as f64 / elapsed.as_secs_f64();

    if !config.quiet {
        println!("engine=rust");
        println!("players={}", config.players.join(","));
        println!("map={}", map_name(config.map_kind));
        println!("parallel={use_parallel}");
    }
    println!("games={}", config.num_games);
    println!("wins={}", stats.wins);
    println!("turns={}", stats.completed_turns);
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!("games_per_second={games_per_second:.2}");
    println!("turns_per_second={turns_per_second:.2}");

    Ok(())
}

fn build_player_templates(codes: &[String], colors: &[Color]) -> Result<Vec<Player>, String> {
    codes
        .iter()
        .zip(colors.iter().copied())
        .map(|(kind, color)| player_from_code(kind, color))
        .collect()
}

fn build_game(config: &Config, player_templates: &[Player], game_index: usize) -> Game {
    let seed = config.seed.map(|base| base.wrapping_add(game_index as u64));
    Game::with_options_and_map_options(
        player_templates.to_vec(),
        seed,
        config.discard_limit,
        config.friendly_robber,
        config.vps_to_win,
        config.map_kind,
        config.number_placement,
    )
}

fn run_tournament(config: &Config, roster: &[String]) -> Result<(), String> {
    if config.jsonl_path.is_some() || config.csv_dir.is_some() {
        return Err("--tournament cannot be combined with --jsonl or --csv".to_string());
    }

    let start = Instant::now();
    let stats = play_tournament(config, roster)?;
    let elapsed = start.elapsed();
    let games_per_second = stats.total_games as f64 / elapsed.as_secs_f64();
    let turns_per_second = stats.completed_turns as f64 / elapsed.as_secs_f64();

    if !config.quiet {
        println!("engine=rust");
        println!("mode=tournament");
        println!("players={}", roster.join(","));
        println!("map={}", map_name(config.map_kind));
        println!("parallel=false");
    }
    println!("tournament_players={}", roster.len());
    println!("games_per_seating={}", config.num_games);
    println!("total_games={}", stats.total_games);
    println!("draws={}", stats.draws);
    println!("turns={}", stats.completed_turns);
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!("games_per_second={games_per_second:.2}");
    println!("turns_per_second={turns_per_second:.2}");
    print_tournament_table(roster, &stats);

    Ok(())
}

fn play_tournament(config: &Config, roster: &[String]) -> Result<TournamentStats, String> {
    let mut stats = TournamentStats {
        scores: vec![TournamentScore::default(); roster.len()],
        ..TournamentStats::default()
    };
    let groups = tournament_groups(roster.len());
    let colors = [Color::Red, Color::Blue, Color::White, Color::Orange];
    let mut global_game_index = 0usize;

    for group in groups {
        for rotation in 0..group.len() {
            let seating = rotated_group(&group, rotation);
            let player_colors = &colors[..seating.len()];
            let player_templates = seating
                .iter()
                .zip(player_colors.iter().copied())
                .map(|(&roster_index, color)| player_from_code(&roster[roster_index], color))
                .collect::<Result<Vec<_>, _>>()?;

            for _ in 0..config.num_games {
                let mut game = build_game(config, &player_templates, global_game_index);
                game.set_record_actions(false);
                game.play();

                stats.total_games += 1;
                stats.completed_turns += game.state.num_turns;
                for &roster_index in &seating {
                    let score = &mut stats.scores[roster_index];
                    score.games += 1;
                    score.turns += game.state.num_turns;
                }

                if let Some(winner) = game.winning_color() {
                    let Some(seat_index) = player_colors.iter().position(|color| *color == winner)
                    else {
                        return Err(format!("winner color {winner:?} was not in seating"));
                    };
                    stats.scores[seating[seat_index]].wins += 1;
                } else {
                    stats.draws += 1;
                    for &roster_index in &seating {
                        stats.scores[roster_index].draws += 1;
                    }
                }

                global_game_index += 1;
            }
        }
    }

    Ok(stats)
}

fn tournament_groups(player_count: usize) -> Vec<Vec<usize>> {
    if player_count <= 4 {
        return vec![(0..player_count).collect()];
    }

    let mut groups = Vec::new();
    for a in 0..player_count - 3 {
        for b in a + 1..player_count - 2 {
            for c in b + 1..player_count - 1 {
                for d in c + 1..player_count {
                    groups.push(vec![a, b, c, d]);
                }
            }
        }
    }
    groups
}

fn rotated_group(group: &[usize], rotation: usize) -> Vec<usize> {
    (0..group.len())
        .map(|index| group[(index + rotation) % group.len()])
        .collect()
}

fn print_tournament_table(roster: &[String], stats: &TournamentStats) {
    let mut rows: Vec<_> = roster
        .iter()
        .zip(stats.scores.iter())
        .map(|(name, score)| {
            let win_rate = if score.games == 0 {
                0.0
            } else {
                score.wins as f64 * 100.0 / score.games as f64
            };
            let draw_rate = if score.games == 0 {
                0.0
            } else {
                score.draws as f64 * 100.0 / score.games as f64
            };
            let avg_turns = if score.games == 0 {
                0.0
            } else {
                score.turns as f64 / score.games as f64
            };
            (name, score, win_rate, draw_rate, avg_turns)
        })
        .collect();
    rows.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| b.1.wins.cmp(&a.1.wins))
            .then_with(|| a.0.cmp(b.0))
    });

    println!("rank,player,games,wins,win_rate,draws,draw_rate,avg_turns");
    for (rank, (name, score, win_rate, draw_rate, avg_turns)) in rows.into_iter().enumerate() {
        println!(
            "{},{},{},{},{:.2},{},{:.2},{:.2}",
            rank + 1,
            name,
            score.games,
            score.wins,
            win_rate,
            score.draws,
            draw_rate,
            avg_turns
        );
    }
}

fn play_one_stats(
    config: &Config,
    player_templates: &[Player],
    game_index: usize,
) -> SimulationStats {
    let mut game = build_game(config, player_templates, game_index);
    game.set_record_actions(false);
    game.play();
    SimulationStats {
        completed_turns: game.state.num_turns,
        wins: usize::from(game.winning_color().is_some()),
    }
}

fn play_parallel_stats(
    config: &Config,
    player_templates: &[Player],
) -> Result<SimulationStats, String> {
    Ok((0..config.num_games)
        .into_par_iter()
        .map(|game_index| play_one_stats(config, player_templates, game_index))
        .reduce(SimulationStats::default, |left, right| SimulationStats {
            completed_turns: left.completed_turns + right.completed_turns,
            wins: left.wins + right.wins,
        }))
}

fn play_serial(
    config: &Config,
    player_colors: &[Color],
    player_templates: &[Player],
    writer: &mut Option<BufWriter<File>>,
    csv_writer: &mut Option<NativeCsvWriter>,
) -> Result<SimulationStats, String> {
    let mut stats = SimulationStats::default();
    for game_index in 0..config.num_games {
        let mut game = build_game(config, player_templates, game_index);
        if writer.is_none() {
            game.set_record_actions(false);
        }
        let records = if csv_writer.is_some() {
            play_collecting_records(
                &mut game,
                game_index,
                player_colors,
                config.map_kind,
                config.include_board_tensor,
            )?
        } else {
            game.play();
            Vec::new()
        };
        if game.winning_color().is_some() {
            stats.wins += 1;
            if let Some(csv_writer) = csv_writer.as_mut() {
                csv_writer.write_game(&game, &records)?;
            }
        }
        stats.completed_turns += game.state.num_turns;

        if let Some(writer) = writer.as_mut() {
            serde_json::to_writer(&mut *writer, &game_to_json_value(&game))
                .map_err(|error| error.to_string())?;
            writeln!(writer).map_err(|error| error.to_string())?;
        }
    }
    Ok(stats)
}

fn play_collecting_records(
    game: &mut Game,
    game_id: usize,
    player_colors: &[Color],
    map_kind: MapKind,
    include_board_tensor: bool,
) -> Result<Vec<NativeRecord>, String> {
    let feature_names = feature_ordering(player_colors.len(), map_kind);
    let action_space = ActionSpace::new(player_colors, map_kind);
    let mut records = Vec::new();
    while game.winning_color().is_none() && game.state.num_turns < TURNS_LIMIT {
        let color = game.state.current_color();
        let state_index = game.state.action_records.len();
        let current_prompt = game.state.current_prompt;
        let legal_action_indices = legal_action_indices_with_space(game, &action_space)?;
        let legal_action_mask =
            legal_action_mask_from_indices(&legal_action_indices, action_space.len());
        let sample = create_sample(game, color);
        let sample_values = feature_names
            .iter()
            .map(|name| sample.get(name).copied().unwrap_or(0.0))
            .collect::<Vec<_>>();
        let board_tensor =
            include_board_tensor.then(|| create_board_tensor_flat(game, color, false).0);
        let action_record = game.play_tick()?;
        let action_index = action_space.index(&action_record.action).ok_or_else(|| {
            format!(
                "action missing from action space: {:?}",
                action_record.action
            )
        })?;
        records.push(NativeRecord {
            game_id,
            step_id: records.len(),
            state_index,
            color,
            current_prompt,
            sample: sample_values,
            action_index,
            action_type_index: action_type_index(action_record.action.action_type),
            legal_action_indices,
            legal_action_mask,
            board_tensor,
        });
    }
    Ok(records)
}

impl NativeCsvWriter {
    fn new(
        path: &Path,
        include_board_tensor: bool,
        num_players: usize,
        map_kind: MapKind,
        player_colors: &[Color],
    ) -> Result<Self, String> {
        create_dir_all(path).map_err(|error| error.to_string())?;
        write_csv_sidecars(
            path,
            include_board_tensor,
            num_players,
            map_kind,
            player_colors,
        )?;
        let board_tensors = if include_board_tensor {
            Some(BufWriter::new(
                File::create(path.join("board_tensors.csv")).map_err(|error| error.to_string())?,
            ))
        } else {
            None
        };
        Ok(Self {
            samples: BufWriter::new(
                File::create(path.join("samples.csv")).map_err(|error| error.to_string())?,
            ),
            actions: BufWriter::new(
                File::create(path.join("actions.csv")).map_err(|error| error.to_string())?,
            ),
            rewards: BufWriter::new(
                File::create(path.join("rewards.csv")).map_err(|error| error.to_string())?,
            ),
            metadata: BufWriter::new(
                File::create(path.join("metadata_rows.csv")).map_err(|error| error.to_string())?,
            ),
            legal_actions: BufWriter::new(
                File::create(path.join("legal_action_indices.csv"))
                    .map_err(|error| error.to_string())?,
            ),
            legal_action_masks: BufWriter::new(
                File::create(path.join("legal_action_masks.csv"))
                    .map_err(|error| error.to_string())?,
            ),
            main: BufWriter::new(
                File::create(path.join("main.csv")).map_err(|error| error.to_string())?,
            ),
            board_tensors,
            feature_names: feature_ordering(num_players, map_kind),
            action_space_size: ActionSpace::new(player_colors, map_kind).len(),
            wrote_header: false,
            include_board_tensor,
        })
    }

    fn write_game(&mut self, game: &Game, records: &[NativeRecord]) -> Result<(), String> {
        if records.is_empty() {
            return Ok(());
        }
        if !self.wrote_header {
            self.write_headers(records)?;
            self.wrote_header = true;
        }
        let rewards = native_returns(game, records);
        let total = records.len();
        let winner = game.winning_color();
        for (index, record) in records.iter().enumerate() {
            let discount = DISCOUNT_FACTOR.powi((total - index - 1) as i32);
            write_csv_row(&mut self.samples, record.sample.iter().copied())?;
            write_csv_row(
                &mut self.actions,
                [record.action_index as f64, record.action_type_index as f64],
            )?;
            write_csv_row(
                &mut self.rewards,
                [
                    rewards[index].0,
                    rewards[index].1,
                    rewards[index].2,
                    rewards[index].0 * discount,
                    rewards[index].1 * discount,
                    rewards[index].2 * discount,
                ],
            )?;
            write_metadata_row(
                &mut self.metadata,
                record,
                index + 1 == total,
                winner,
                game.state.num_turns,
            )?;
            write_legal_action_indices(&mut self.legal_actions, &record.legal_action_indices)?;
            write_legal_action_mask(&mut self.legal_action_masks, &record.legal_action_mask)?;
            if let Some(writer) = self.board_tensors.as_mut()
                && let Some(tensor) = &record.board_tensor
            {
                write_csv_row(writer, tensor.iter().copied())?;
            }
            write_csv_row(
                &mut self.main,
                record
                    .sample
                    .iter()
                    .copied()
                    .chain(
                        record
                            .board_tensor
                            .as_ref()
                            .into_iter()
                            .flat_map(|tensor| tensor.iter().copied()),
                    )
                    .chain([record.action_index as f64, record.action_type_index as f64])
                    .chain([
                        rewards[index].0,
                        rewards[index].1,
                        rewards[index].2,
                        rewards[index].0 * discount,
                        rewards[index].1 * discount,
                        rewards[index].2 * discount,
                    ]),
            )?;
        }
        Ok(())
    }

    fn write_headers(&mut self, records: &[NativeRecord]) -> Result<(), String> {
        write_csv_header(
            &mut self.samples,
            self.feature_names.iter().map(|name| format!("F_{name}")),
        )?;
        write_csv_header(
            &mut self.actions,
            ["ACTION".to_string(), "ACTION_TYPE".to_string()],
        )?;
        let reward_columns = [
            "RETURN",
            "TOURNAMENT_RETURN",
            "VICTORY_POINTS_RETURN",
            "DISCOUNTED_RETURN",
            "DISCOUNTED_TOURNAMENT_RETURN",
            "DISCOUNTED_VICTORY_POINTS_RETURN",
        ];
        write_csv_header(
            &mut self.rewards,
            reward_columns.iter().map(|name| name.to_string()),
        )?;
        write_csv_header(
            &mut self.metadata,
            [
                "GAME_ID",
                "STEP_ID",
                "STATE_INDEX",
                "COLOR",
                "CURRENT_PROMPT",
                "DONE",
                "WINNER",
                "NUM_TURNS",
            ]
            .into_iter()
            .map(str::to_string),
        )?;
        writeln!(self.legal_actions, "LEGAL_ACTION_INDICES").map_err(|error| error.to_string())?;
        write_csv_header(
            &mut self.legal_action_masks,
            (0..self.action_space_size).map(|index| format!("A_{index}")),
        )?;
        if let Some(writer) = self.board_tensors.as_mut() {
            let width = records
                .iter()
                .find_map(|record| record.board_tensor.as_ref().map(Vec::len))
                .unwrap_or(0);
            write_csv_header(writer, (0..width).map(|index| format!("BT_{index}")))?;
        }
        let board_width = if self.include_board_tensor {
            records
                .iter()
                .find_map(|record| record.board_tensor.as_ref().map(Vec::len))
                .unwrap_or(0)
        } else {
            0
        };
        write_csv_header(
            &mut self.main,
            self.feature_names
                .iter()
                .map(|name| format!("F_{name}"))
                .chain((0..board_width).map(|index| format!("BT_{index}")))
                .chain(["ACTION".to_string(), "ACTION_TYPE".to_string()])
                .chain(reward_columns.iter().map(|name| name.to_string())),
        )
    }

    fn flush(&mut self) -> Result<(), String> {
        self.samples.flush().map_err(|error| error.to_string())?;
        self.actions.flush().map_err(|error| error.to_string())?;
        self.rewards.flush().map_err(|error| error.to_string())?;
        self.metadata.flush().map_err(|error| error.to_string())?;
        self.legal_actions
            .flush()
            .map_err(|error| error.to_string())?;
        self.legal_action_masks
            .flush()
            .map_err(|error| error.to_string())?;
        self.main.flush().map_err(|error| error.to_string())?;
        if let Some(writer) = self.board_tensors.as_mut() {
            writer.flush().map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

fn native_returns(game: &Game, records: &[NativeRecord]) -> Vec<(f64, f64, f64)> {
    let winning_color = game.winning_color();
    let colors = records
        .iter()
        .map(|record| record.color)
        .collect::<BTreeSet<_>>();
    let final_samples = colors
        .into_iter()
        .map(|color| {
            let sample = create_sample(game, color);
            let points = sample
                .get("P0_ACTUAL_VPS")
                .copied()
                .unwrap_or(0.0)
                .min(10.0);
            (color, points)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let turn_discount = 0.9999_f64.powi(game.state.num_turns as i32);
    records
        .iter()
        .map(|record| {
            let sign = if winning_color == Some(record.color) {
                1.0
            } else {
                -1.0
            };
            let points = final_samples.get(&record.color).copied().unwrap_or(0.0);
            (
                sign,
                sign * 1000.0 + points * turn_discount,
                points * turn_discount,
            )
        })
        .collect()
}

fn write_csv_header(
    writer: &mut BufWriter<File>,
    columns: impl IntoIterator<Item = String>,
) -> Result<(), String> {
    let mut first = true;
    for column in columns {
        if !first {
            write!(writer, ",").map_err(|error| error.to_string())?;
        }
        first = false;
        write!(writer, "{column}").map_err(|error| error.to_string())?;
    }
    writeln!(writer).map_err(|error| error.to_string())
}

fn write_csv_row(
    writer: &mut BufWriter<File>,
    values: impl IntoIterator<Item = f64>,
) -> Result<(), String> {
    let mut first = true;
    for value in values {
        if !first {
            write!(writer, ",").map_err(|error| error.to_string())?;
        }
        first = false;
        write!(writer, "{value}").map_err(|error| error.to_string())?;
    }
    writeln!(writer).map_err(|error| error.to_string())
}

fn write_csv_sidecars(
    path: &Path,
    include_board_tensor: bool,
    num_players: usize,
    map_kind: MapKind,
    player_colors: &[Color],
) -> Result<(), String> {
    let feature_names = feature_ordering(num_players, map_kind);
    let action_space = ActionSpace::new(player_colors, map_kind).json_value();
    let metadata = serde_json::json!({
        "format": "catanatron-rs-native-csv-v1",
        "files": {
            "samples": "samples.csv",
            "actions": "actions.csv",
            "rewards": "rewards.csv",
            "metadata": "metadata_rows.csv",
            "legal_action_indices": "legal_action_indices.csv",
            "legal_action_masks": "legal_action_masks.csv",
            "main": "main.csv",
            "board_tensors": include_board_tensor.then_some("board_tensors.csv"),
        },
        "map": map_name(map_kind),
        "num_players": num_players,
        "player_colors": player_colors.iter().map(|color| color_index(*color)).collect::<Vec<_>>(),
        "discount_factor": DISCOUNT_FACTOR,
        "board_tensor_shape": include_board_tensor.then_some(board_tensor_shape(num_players, false)),
        "action_space_size": action_space.as_array().map(Vec::len).unwrap_or(0),
        "reward_columns": [
            "RETURN",
            "TOURNAMENT_RETURN",
            "VICTORY_POINTS_RETURN",
            "DISCOUNTED_RETURN",
            "DISCOUNTED_TOURNAMENT_RETURN",
            "DISCOUNTED_VICTORY_POINTS_RETURN",
        ],
    });
    write_json_file(path.join("metadata.json"), &metadata)?;
    write_json_file(
        path.join("feature_ordering.json"),
        &serde_json::json!(feature_names),
    )?;
    write_json_file(path.join("action_space.json"), &action_space)?;
    Ok(())
}

fn write_json_file(path: PathBuf, value: &serde_json::Value) -> Result<(), String> {
    let writer = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
    serde_json::to_writer_pretty(writer, value).map_err(|error| error.to_string())
}

fn write_metadata_row(
    writer: &mut BufWriter<File>,
    record: &NativeRecord,
    done: bool,
    winner: Option<Color>,
    num_turns: usize,
) -> Result<(), String> {
    writeln!(
        writer,
        "{},{},{},{},{},{},{},{}",
        record.game_id,
        record.step_id,
        record.state_index,
        color_index(record.color),
        action_prompt_index(record.current_prompt),
        u8::from(done),
        winner.map(color_index).map_or(-1, i16::from),
        num_turns
    )
    .map_err(|error| error.to_string())
}

fn write_legal_action_indices(
    writer: &mut BufWriter<File>,
    legal_action_indices: &[usize],
) -> Result<(), String> {
    for (index, action_index) in legal_action_indices.iter().enumerate() {
        if index > 0 {
            write!(writer, ";").map_err(|error| error.to_string())?;
        }
        write!(writer, "{action_index}").map_err(|error| error.to_string())?;
    }
    writeln!(writer).map_err(|error| error.to_string())
}

fn write_legal_action_mask(
    writer: &mut BufWriter<File>,
    legal_action_mask: &[u8],
) -> Result<(), String> {
    let mut first = true;
    for value in legal_action_mask {
        if !first {
            write!(writer, ",").map_err(|error| error.to_string())?;
        }
        first = false;
        write!(writer, "{value}").map_err(|error| error.to_string())?;
    }
    writeln!(writer).map_err(|error| error.to_string())
}

fn color_index(color: Color) -> u8 {
    match color {
        Color::Red => 0,
        Color::Blue => 1,
        Color::Orange => 2,
        Color::White => 3,
    }
}

fn action_prompt_index(prompt: ActionPrompt) -> u8 {
    match prompt {
        ActionPrompt::BuildInitialSettlement => 0,
        ActionPrompt::BuildInitialRoad => 1,
        ActionPrompt::PlayTurn => 2,
        ActionPrompt::Discard => 3,
        ActionPrompt::MoveRobber => 4,
        ActionPrompt::DecideTrade => 5,
        ActionPrompt::DecideAcceptees => 6,
    }
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut config = Config::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-n" | "--num" => {
                config.num_games = parse_next(&mut args, "--num")?;
                if config.num_games == 0 {
                    return Err("--num must be greater than 0".to_string());
                }
            }
            "--players" => {
                let value: String = require_next(&mut args, "--players")?;
                config.players = parse_player_list(&value);
                if !(2..=4).contains(&config.players.len()) {
                    return Err("--players must include 2 to 4 comma-separated players".to_string());
                }
            }
            "--tournament" => {
                let value: String = require_next(&mut args, "--tournament")?;
                let roster = parse_player_list(&value);
                if roster.len() < 2 {
                    return Err("--tournament must include at least 2 players".to_string());
                }
                let unique: BTreeSet<_> = roster.iter().collect();
                if unique.len() != roster.len() {
                    return Err("--tournament player names must be unique".to_string());
                }
                config.tournament_players = Some(roster);
            }
            "--seed" => config.seed = Some(parse_next(&mut args, "--seed")?),
            "--map" => {
                let value = require_next::<String>(&mut args, "--map")?.to_uppercase();
                config.map_kind = match value.as_str() {
                    "BASE" => MapKind::Base,
                    "TOURNAMENT" => MapKind::Tournament,
                    "MINI" => MapKind::Mini,
                    other => {
                        return Err(format!(
                            "--map must be BASE, MINI, or TOURNAMENT; got {other}"
                        ));
                    }
                }
            }
            "--number-placement" => {
                let value = require_next::<String>(&mut args, "--number-placement")?.to_lowercase();
                config.number_placement = match value.as_str() {
                    "official_spiral" => NumberPlacement::OfficialSpiral,
                    "random" => NumberPlacement::Random,
                    other => {
                        return Err(format!(
                            "--number-placement must be official_spiral or random; got {other}"
                        ));
                    }
                };
            }
            "--discard-limit" => {
                config.discard_limit = parse_next(&mut args, "--discard-limit")?;
                if !(5..=20).contains(&config.discard_limit) {
                    return Err("--discard-limit must be between 5 and 20".to_string());
                }
            }
            "--friendly-robber" => config.friendly_robber = true,
            "--vps-to-win" => {
                config.vps_to_win = parse_next(&mut args, "--vps-to-win")?;
                if !(1..=20).contains(&config.vps_to_win) {
                    return Err("--vps-to-win must be between 1 and 20".to_string());
                }
            }
            "--jsonl" | "--output" => {
                config.jsonl_path = Some(PathBuf::from(require_next::<String>(&mut args, &arg)?));
            }
            "--csv" => {
                config.csv_dir = Some(PathBuf::from(require_next::<String>(&mut args, &arg)?));
            }
            "--include-board-tensor" => config.include_board_tensor = true,
            "--no-parallel" => config.parallel = false,
            "-q" | "--quiet" => config.quiet = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }
    Ok(config)
}

fn parse_player_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|part| part.trim().to_uppercase())
        .filter(|part| !part.is_empty())
        .collect()
}

fn require_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse()
        .map_err(|_| format!("invalid value for {flag}"))
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    require_next(args, flag)
}

fn player_from_code(code: &str, color: Color) -> Result<Player, String> {
    let code = normalized_player_code(code);
    if let Some(depth) = player_code_param(&code, &["AB", "ALPHABETA", "ALPHABETAN"]) {
        return Ok(Player::alpha_beta(color, depth));
    }
    if let Some(depth) =
        player_code_param(&code, &["SAB", "SAMETURNALPHABETA", "SAMETURNALPHABETAN"])
    {
        return Ok(Player::same_turn_alpha_beta(color, depth));
    }
    if let Some(depth) = player_code_param(
        &code,
        &[
            "HYBRID",
            "HYBRIDAB",
            "STRATEGICAB",
            "STRATEGICALPHABETA",
            "STRATEGICALPHABETAN",
        ],
    ) {
        return Ok(Player::strategic_alpha_beta(color, depth));
    }
    if let Some(playouts) = player_code_param(
        &code,
        &[
            "P",
            "PLAYOUT",
            "PLAYOUTN",
            "GREEDYPLAYOUTS",
            "GREEDYPLAYOUTSN",
            "MCTS",
            "MCTSN",
        ],
    ) {
        return Ok(Player::playout(color, playouts.into()));
    }
    if let Some(playouts) = player_code_param(
        &code,
        &[
            "SP",
            "STRATEGICPLAYOUT",
            "STRATEGICPLAYOUTN",
            "HYBRIDROLL",
            "HYBRIDROLLOUT",
            "IMPROVED",
        ],
    ) {
        return Ok(Player::strategic_playout(color, playouts.into()));
    }

    match code.as_str() {
        "S" | "SIMPLE" => Ok(Player::simple(color)),
        "R" | "RANDOM" => Ok(Player::random(color)),
        "W" | "WEIGHTEDRANDOM" => Ok(Player::weighted_random(color)),
        "VP" | "VICTORYPOINT" => Ok(Player::victory_point(color)),
        "F" | "VALUE" | "VALUEFUNCTION" => Ok(Player::value_function(color)),
        "SV" | "STRATEGIC" | "STRATEGICVALUE" => Ok(Player::strategic_value(color)),
        "HYBRID" | "HYBRIDAB" | "STRATEGICAB" | "STRATEGICALPHABETA" => {
            Ok(Player::strategic_alpha_beta(color, 2))
        }
        "IMPROVED" | "HYBRIDROLL" | "HYBRIDROLLOUT" | "STRATEGICPLAYOUT" => {
            Ok(Player::strategic_playout(color, 16))
        }
        "ENSEMBLE" | "CHAMP" => Ok(Player::ensemble(color)),
        "JSETTLERS" | "JSETTLERS2" => Ok(Player::value_function(color)),
        "QSETTLERS" | "DQN" | "CS7641QSETTLERS" => Ok(Player::value_function(color)),
        "AB" | "CATANATRON" | "CATANATRONBOT" | "ALPHABETA" => Ok(Player::alpha_beta(color, 2)),
        "SAB" | "SAMETURNALPHABETA" => Ok(Player::same_turn_alpha_beta(color, 2)),
        "P" | "PLAYOUT" => Ok(Player::playout(color, 8)),
        "GREEDYPLAYOUTS" | "GREEDYPLAYOUT" | "GP" => Ok(Player::playout(color, 25)),
        "MCTS" => Ok(Player::playout(color, 100)),
        "RLCATAN" | "RL" | "SETTLERSRL" | "HENRYCHARLESWORTH" => Ok(Player::playout(color, 1)),
        "CATANAI" | "CATANRL" | "KVOMBATKERE" => Ok(Player::playout(color, 2)),
        "BOTAN" | "SETTLERSOFBOTAN" => Ok(Player::victory_point(color)),
        "ZARNS" | "ZARNSMCTS" => Ok(Player::playout(color, 1)),
        "PYCATAN" => Ok(Player::weighted_random(color)),
        "MONTECATANO" | "MONTE" | "ALGORYTHMSXV" => Ok(Player::playout(color, 25)),
        "SMARTSETTLERS" | "SMARTSETTLER" => Ok(Player::playout(color, 25)),
        "HENRYHORSE" | "HENRYHORSECATAN" => Ok(Player::playout(color, 1)),
        "JUSTINASHER" | "JUSTINCASHER" | "JUSTINASHERSELFPLAY" => Ok(Player::value_function(color)),
        "RASMUSGREVE" | "RASMUSGREVECATAN" => Ok(Player::weighted_random(color)),
        "HRODGEIR" | "OPENINGAI" | "STARTINGAI" => Ok(Player::victory_point(color)),
        "MATTYB5722" | "LINEARREGRESSIONAI" => Ok(Player::weighted_random(color)),
        other => Err(format!("unsupported player kind: {other}")),
    }
}

fn normalized_player_code(code: &str) -> String {
    code.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_uppercase())
        .collect()
}

fn player_code_param(code: &str, prefixes: &[&str]) -> Option<u8> {
    prefixes.iter().find_map(|prefix| {
        let suffix = code.strip_prefix(prefix)?;
        if suffix.is_empty() || !suffix.chars().all(|ch| ch.is_ascii_digit()) {
            return None;
        }
        suffix.parse::<u8>().ok().filter(|value| *value > 0)
    })
}

fn output_path(path: &Path) -> Result<PathBuf, String> {
    if path.extension().and_then(|value| value.to_str()) == Some("jsonl") {
        return Ok(path.to_path_buf());
    }
    create_dir_all(path).map_err(|error| error.to_string())?;
    Ok(path.join("rust-games.jsonl"))
}

fn map_name(map_kind: MapKind) -> &'static str {
    match map_kind {
        MapKind::Base => "BASE",
        MapKind::Mini => "MINI",
        MapKind::Tournament => "TOURNAMENT",
    }
}

fn print_help() {
    println!(
        "\
catanatron native Rust simulator

Usage:
  catanatron --num 1000 --players R,R,R,R --quiet

Options:
  -n, --num <N>              Number of games to simulate
      --players <P,...>      2-4 player kinds: R,W,VP,F,AB,SAB,P
      --tournament <P,...>   Multi-agent tournament. For >4 players, evaluates
                             every 4-player pod with seat rotations.
                             Catanatron aliases include AlphaBeta(n=2),
                             ValueFunction, GreedyPlayouts(n=25), MCTS(n=100).
                             External aliases include JSETTLERS, QSETTLERS,
                             RLCATAN, CATANAI, BOTAN, MONTECATANO, SMARTSETTLERS
      --seed <N>             Base seed; each game adds its index
      --map <BASE|MINI|TOURNAMENT>
                              Map template
      --number-placement <official_spiral|random>
                              Number placement strategy
      --discard-limit <N>    Discard threshold, 5-20
      --friendly-robber      Enable friendly robber
      --vps-to-win <N>       Victory points to win, 1-20
      --jsonl <PATH>         Write final game snapshots as JSONL
      --csv <DIR>            Write native CSV training matrices
      --include-board-tensor Include board tensor columns in native CSV output
      --no-parallel          Disable Rayon parallel stats mode
  -q, --quiet                Print only stats
"
    );
}

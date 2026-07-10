use catanatron_rs::{
    ActionSpace, Color, Game, MapKind, NumberPlacement, Player, action_space_json_value,
    board_tensor_flat_len, create_board_tensor_batch_flat, create_board_tensor_flat,
    feature_ordering, fill_board_tensor_batch_flat_f32, fill_board_tensor_flat,
    fill_board_tensor_flat_f32, fill_sample_vector_batch_for_schema, generate_playable_actions,
    legal_action_mask_with_space,
};
use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};

fn random_players(count: usize) -> Vec<Player> {
    Color::ALL
        .into_iter()
        .take(count)
        .map(Player::random)
        .collect()
}

fn prebuilt_training_games(batch_size: usize) -> Vec<Game> {
    (0..batch_size)
        .map(|index| {
            let mut game = Game::with_options(random_players(4), Some(index as u64), 7, false, 10);
            for _ in 0..[0usize, 12, 40, 80][index % 4] {
                if game.winning_color().is_some() {
                    break;
                }
                let _ = game.play_tick();
            }
            game
        })
        .collect()
}

fn bench_full_games(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_game");
    group.sample_size(20);
    for player_count in [2usize, 4] {
        group.bench_with_input(
            BenchmarkId::new("base_random_players", player_count),
            &player_count,
            |b, &player_count| {
                b.iter_batched(
                    || Game::with_options(random_players(player_count), Some(1), 7, false, 10),
                    |mut game| {
                        let winner = game.play();
                        (winner, game.state.num_turns)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.bench_function("tournament_random_numbers_4p", |b| {
        b.iter_batched(
            || {
                Game::with_options_and_map_options(
                    random_players(4),
                    Some(1),
                    7,
                    false,
                    10,
                    MapKind::Tournament,
                    NumberPlacement::Random,
                )
            },
            |mut game| {
                let winner = game.play();
                (winner, game.state.num_turns)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_action_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("actions");
    group.bench_function("initial_playable_actions_4p", |b| {
        b.iter_batched(
            || Game::with_options(random_players(4), Some(3), 7, false, 10),
            |game| generate_playable_actions(&game.state),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("static_action_space_base_4p", |b| {
        b.iter(|| action_space_json_value(&Color::ALL, MapKind::Base));
    });
    group.bench_function("legal_action_mask_midgame_4p", |b| {
        b.iter_batched(
            || {
                let mut game = Game::with_options(random_players(4), Some(17), 7, false, 10);
                for _ in 0..40 {
                    if game.winning_color().is_some() {
                        break;
                    }
                    let _ = game.play_tick();
                }
                let action_space = ActionSpace::new(&Color::ALL, MapKind::Base);
                (game, action_space)
            },
            |(game, action_space)| legal_action_mask_with_space(&game, &action_space).unwrap(),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_board_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("board_tensor");
    group.bench_function("board_tensor_flat_4p_channels_last", |b| {
        b.iter_batched(
            || {
                let mut game = Game::with_options(random_players(4), Some(5), 7, false, 10);
                for _ in 0..12 {
                    let _ = game.play_tick();
                }
                game
            },
            |game| create_board_tensor_flat(&game, Color::Red, false),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("board_tensor_fill_4p_channels_last", |b| {
        b.iter_batched(
            || {
                let mut game = Game::with_options(random_players(4), Some(5), 7, false, 10);
                for _ in 0..12 {
                    let _ = game.play_tick();
                }
                let out = vec![0.0; board_tensor_flat_len(4)];
                (game, out)
            },
            |(game, mut out)| {
                fill_board_tensor_flat(&game, Color::Red, false, &mut out).unwrap();
                out
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("board_tensor_fill_f32_4p_channels_last", |b| {
        b.iter_batched(
            || {
                let mut game = Game::with_options(random_players(4), Some(5), 7, false, 10);
                for _ in 0..12 {
                    let _ = game.play_tick();
                }
                let out = vec![0.0_f32; board_tensor_flat_len(4)];
                (game, out)
            },
            |(game, mut out)| {
                fill_board_tensor_flat_f32(&game, Color::Red, false, &mut out).unwrap();
                out
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("board_tensor_batch_4p_channels_last_32", |b| {
        b.iter_batched(
            || {
                (0..32)
                    .map(|index| {
                        let mut game =
                            Game::with_options(random_players(4), Some(index), 7, false, 10);
                        for _ in 0..12 {
                            let _ = game.play_tick();
                        }
                        game
                    })
                    .collect::<Vec<_>>()
            },
            |games| {
                let samples = games
                    .iter()
                    .map(|game| (game, game.state.current_color()))
                    .collect::<Vec<_>>();
                create_board_tensor_batch_flat(&samples, false)
            },
            BatchSize::SmallInput,
        );
    });
    for batch_size in [64usize, 256, 1024] {
        let prebuilt_games = prebuilt_training_games(batch_size);
        let prebuilt_samples = prebuilt_games
            .iter()
            .map(|game| (game, game.state.current_color()))
            .collect::<Vec<_>>();
        let row_len = board_tensor_flat_len(4);
        group.throughput(Throughput::Bytes((batch_size * row_len * 4) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "board_tensor_batch_f32_fill_4p_channels_last_prebuilt",
                batch_size,
            ),
            &batch_size,
            |b, _| {
                b.iter_batched_ref(
                    || vec![0.0_f32; row_len * batch_size],
                    |out| {
                        fill_board_tensor_batch_flat_f32(
                            black_box(&prebuilt_samples),
                            false,
                            black_box(out),
                        )
                        .unwrap()
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_training_native(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_native");
    group.sample_size(10);
    for batch_size in [64usize, 256, 1024] {
        let prebuilt_games = prebuilt_training_games(batch_size);
        let prebuilt_samples = prebuilt_games
            .iter()
            .map(|game| (game, game.state.current_color()))
            .collect::<Vec<_>>();
        let schema = feature_ordering(4, MapKind::Base);
        let row_len = schema.len();
        group.throughput(Throughput::Bytes((batch_size * row_len * 4) as u64));
        group.bench_with_input(
            BenchmarkId::new("feature_vectors_f32_fill_4p_prebuilt_reused", batch_size),
            &batch_size,
            |b, _| {
                let mut out = vec![0.0_f32; row_len * batch_size];
                b.iter(|| {
                    fill_sample_vector_batch_for_schema(
                        black_box(&prebuilt_samples),
                        black_box(&schema),
                        black_box(out.as_mut_slice()),
                    )
                    .unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_full_games,
    bench_action_generation,
    bench_board_tensor,
    bench_training_native
);
criterion_main!(benches);

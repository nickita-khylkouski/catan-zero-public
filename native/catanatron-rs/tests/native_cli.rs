use std::fs;
use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_catanatron")
}

#[test]
fn native_cli_prints_simulation_stats() {
    let output = Command::new(binary())
        .args(["--num", "2", "--players", "R,R", "--seed", "1", "--quiet"])
        .output()
        .expect("native CLI should run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("games=2"));
    assert!(stdout.contains("wins="));
    assert!(stdout.contains("turns="));
    assert!(stdout.contains("games_per_second="));
}

#[test]
fn native_cli_accepts_random_number_placement() {
    let output = Command::new(binary())
        .args([
            "--num",
            "2",
            "--players",
            "R,R",
            "--seed",
            "1",
            "--number-placement",
            "random",
            "--quiet",
        ])
        .output()
        .expect("native CLI should run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("games=2"));
}

#[test]
fn native_cli_accepts_external_ai_aliases() {
    let output = Command::new(binary())
        .args([
            "--num",
            "1",
            "--players",
            "AlphaBeta(n=2),GreedyPlayouts(n=1)",
            "--seed",
            "1",
            "--vps-to-win",
            "2",
            "--quiet",
        ])
        .output()
        .expect("native CLI should run");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("games=1"));
    assert!(stdout.contains("wins="));
}

#[test]
fn native_cli_runs_external_ai_tournament() {
    let output = Command::new(binary())
        .args([
            "--num",
            "1",
            "--tournament",
            "JSETTLERS,AB2,QSETTLERS,RLCATAN,CATANAI,BOTAN,ZARNS,MONTECATANO",
            "--seed",
            "2",
            "--vps-to-win",
            "2",
            "--quiet",
        ])
        .output()
        .expect("native CLI should run");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("tournament_players=8"));
    assert!(stdout.contains("total_games=280"));
    assert!(stdout.contains("rank,player,games,wins,win_rate,draws,draw_rate,avg_turns"));
    assert!(stdout.contains("JSETTLERS"));
    assert!(stdout.contains("MONTECATANO"));
}

#[test]
fn native_cli_accepts_tournament_map() {
    let output = Command::new(binary())
        .args([
            "--num",
            "2",
            "--players",
            "R,R",
            "--seed",
            "1",
            "--map",
            "TOURNAMENT",
            "--number-placement",
            "random",
            "--quiet",
        ])
        .output()
        .expect("native CLI should run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("games=2"));
}

#[test]
fn native_cli_parallel_stats_match_serial_seeded_run() {
    let common_args = [
        "--num",
        "8",
        "--players",
        "R,R",
        "--seed",
        "11",
        "--vps-to-win",
        "3",
        "--quiet",
    ];
    let parallel = Command::new(binary())
        .args(common_args)
        .output()
        .expect("parallel native CLI should run");
    let serial = Command::new(binary())
        .args(common_args)
        .arg("--no-parallel")
        .output()
        .expect("serial native CLI should run");

    assert!(parallel.status.success());
    assert!(serial.status.success());
    let parallel_stdout = String::from_utf8(parallel.stdout).expect("parallel stdout utf8");
    let serial_stdout = String::from_utf8(serial.stdout).expect("serial stdout utf8");
    for key in ["games", "wins", "turns"] {
        assert_eq!(
            stat_value(&parallel_stdout, key),
            stat_value(&serial_stdout, key),
            "{key} should match"
        );
    }
}

#[test]
fn native_cli_writes_jsonl_snapshots() {
    let path = std::env::temp_dir().join(format!(
        "catanatron-native-cli-{}-{}.jsonl",
        std::process::id(),
        unique_suffix()
    ));
    let output = Command::new(binary())
        .args([
            "--num",
            "2",
            "--players",
            "R,R",
            "--seed",
            "2",
            "--vps-to-win",
            "3",
            "--jsonl",
            path.to_str().expect("temp path should be utf8"),
            "--quiet",
        ])
        .output()
        .expect("native CLI should run");

    assert!(output.status.success());
    let contents = fs::read_to_string(&path).expect("jsonl should be written");
    let lines = contents.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    for line in lines {
        let snapshot: serde_json::Value = serde_json::from_str(line).expect("valid json line");
        assert!(snapshot["state_index"].as_u64().unwrap() > 0);
        assert!(snapshot["tiles"].as_array().unwrap().len() >= 7);
        assert!(snapshot.as_object().unwrap().contains_key("winning_color"));
    }
    let _ = fs::remove_file(path);
}

#[test]
fn native_cli_writes_csv_matrices_with_board_tensors() {
    let path = std::env::temp_dir().join(format!(
        "catanatron-native-cli-csv-{}-{}",
        std::process::id(),
        unique_suffix()
    ));
    let output = Command::new(binary())
        .args([
            "--num",
            "1",
            "--players",
            "R,R",
            "--seed",
            "3",
            "--vps-to-win",
            "3",
            "--csv",
            path.to_str().expect("temp path should be utf8"),
            "--include-board-tensor",
            "--quiet",
        ])
        .output()
        .expect("native CLI should run");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for filename in [
        "samples.csv",
        "actions.csv",
        "rewards.csv",
        "metadata_rows.csv",
        "legal_action_indices.csv",
        "legal_action_masks.csv",
        "main.csv",
        "board_tensors.csv",
        "metadata.json",
        "feature_ordering.json",
        "action_space.json",
    ] {
        assert!(path.join(filename).exists(), "{filename} should be written");
    }

    let samples = fs::read_to_string(path.join("samples.csv")).expect("samples csv");
    let actions = fs::read_to_string(path.join("actions.csv")).expect("actions csv");
    let rewards = fs::read_to_string(path.join("rewards.csv")).expect("rewards csv");
    let metadata_rows =
        fs::read_to_string(path.join("metadata_rows.csv")).expect("metadata rows csv");
    let legal_actions =
        fs::read_to_string(path.join("legal_action_indices.csv")).expect("legal actions csv");
    let legal_action_masks =
        fs::read_to_string(path.join("legal_action_masks.csv")).expect("legal action masks csv");
    let board_tensors =
        fs::read_to_string(path.join("board_tensors.csv")).expect("board tensors csv");
    let metadata: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(path.join("metadata.json")).expect("metadata json"),
    )
    .expect("valid metadata json");
    let feature_ordering: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(path.join("feature_ordering.json")).expect("feature ordering json"),
    )
    .expect("valid feature ordering json");
    let action_space: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(path.join("action_space.json")).expect("action space json"),
    )
    .expect("valid action space json");

    assert!(samples.lines().next().unwrap().contains("F_P0_ACTUAL_VPS"));
    assert_eq!(actions.lines().next().unwrap(), "ACTION,ACTION_TYPE");
    assert!(
        rewards
            .lines()
            .next()
            .unwrap()
            .contains("DISCOUNTED_RETURN")
    );
    assert_eq!(
        metadata_rows.lines().next().unwrap(),
        "GAME_ID,STEP_ID,STATE_INDEX,COLOR,CURRENT_PROMPT,DONE,WINNER,NUM_TURNS"
    );
    assert_eq!(
        legal_actions.lines().next().unwrap(),
        "LEGAL_ACTION_INDICES"
    );
    let action_space_size = metadata["action_space_size"].as_u64().unwrap() as usize;
    assert_eq!(
        legal_action_masks
            .lines()
            .next()
            .unwrap()
            .split(',')
            .count(),
        action_space_size
    );
    assert_eq!(metadata["format"], "catanatron-rs-native-csv-v1");
    assert_eq!(metadata["num_players"], 2);
    assert_eq!(metadata["files"]["board_tensors"], "board_tensors.csv");
    assert_eq!(
        metadata["files"]["legal_action_masks"],
        "legal_action_masks.csv"
    );
    assert_eq!(
        metadata["board_tensor_shape"],
        serde_json::json!([21, 11, 16])
    );
    assert!(feature_ordering.as_array().unwrap().len() > 100);
    assert!(action_space.as_array().unwrap().len() > 100);
    assert_eq!(
        board_tensors.lines().next().unwrap().split(',').count(),
        21 * 11 * 16
    );
    assert!(samples.lines().count() > 1);
    assert_eq!(samples.lines().count(), actions.lines().count());
    assert_eq!(samples.lines().count(), rewards.lines().count());
    assert_eq!(samples.lines().count(), metadata_rows.lines().count());
    assert_eq!(samples.lines().count(), legal_actions.lines().count());
    assert_eq!(samples.lines().count(), legal_action_masks.lines().count());
    assert_eq!(samples.lines().count(), board_tensors.lines().count());

    let sparse_rows = legal_actions.lines().skip(1);
    let dense_rows = legal_action_masks.lines().skip(1);
    for (sparse, dense) in sparse_rows.zip(dense_rows) {
        let sparse_count = sparse.split(';').filter(|value| !value.is_empty()).count();
        let dense_values = dense.split(',').collect::<Vec<_>>();
        assert_eq!(dense_values.len(), action_space_size);
        assert!(
            dense_values
                .iter()
                .all(|value| *value == "0" || *value == "1")
        );
        assert_eq!(
            dense_values.iter().filter(|value| **value == "1").count(),
            sparse_count
        );
    }

    let _ = fs::remove_dir_all(path);
}

fn stat_value<'a>(stdout: &'a str, key: &str) -> &'a str {
    let prefix = format!("{key}=");
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("{key} should be present in stdout:\n{stdout}"))
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos()
}

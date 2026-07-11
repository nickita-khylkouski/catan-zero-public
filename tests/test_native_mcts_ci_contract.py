"""Static guard that keeps native MCTS CI executable rather than skippable."""

from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "native-mcts.yml"
)


def test_native_ci_builds_cp311_wheel_and_asserts_symbols() -> None:
    source = WORKFLOW.read_text()

    assert 'python-version: "3.11"' in source
    assert "maturin build --locked --release" in source
    assert "native/catanatron-rs/python/Cargo.toml" in source
    assert "dist-ci/catanatron_rs-*-cp311-*.whl" in source
    assert 'getattr(catanatron_rs, "gumbel_search", None)' in source
    assert 'getattr(catanatron_rs, "build_entity_features_flat", None)' in source
    assert "tests/test_native_gumbel_hot_loop.py" in source
    assert "tests/test_native_information_set_search.py" in source
    assert "tests/test_generate_information_set_invariants.py" in source
    assert "tests/test_generate_native_rollout.py" in source


def test_native_ci_has_non_skipping_rust_gates() -> None:
    source = WORKFLOW.read_text()

    for command in (
        "cargo fmt --manifest-path native/gumbel_mcts_rs/Cargo.toml",
        "cargo check --locked --manifest-path native/gumbel_mcts_rs/Cargo.toml",
        "cargo test --locked --manifest-path native/gumbel_mcts_rs/Cargo.toml",
        "cargo clippy --locked --manifest-path native/gumbel_mcts_rs/Cargo.toml --all-targets -- -D warnings",
        "cargo clippy --locked --manifest-path native/catanatron-rs/python/Cargo.toml --all-targets -- -D warnings",
        "cargo check --locked --manifest-path native/catanatron-rs/python/Cargo.toml",
        "cargo test --locked --manifest-path native/catanatron-rs/Cargo.toml",
    ):
        assert command in source

"""Native wheel inputs must be present in newly rendered runtime closures."""

import json
from pathlib import Path

import pytest

from tools import a1_pre_wave_contract as contract


def test_runtime_closure_binds_complete_native_mcts_inputs() -> None:
    paths = {
        Path(record["path"]).as_posix()
        for record in contract._runtime_code_tree_records()
    }

    for suffix in (
        "native/gumbel_mcts_rs/Cargo.lock",
        "native/gumbel_mcts_rs/src/lib.rs",
        "native/gumbel_mcts_rs/src/python_binding.rs",
    ):
        assert any(path.endswith(suffix) for path in paths), suffix


def test_future_contract_and_fleet_launcher_require_native_rollout() -> None:
    root = Path(__file__).resolve().parents[1]
    template = json.loads(
        (root / "configs/experiments/a1_pre_wave_contract.template.json").read_text()
    )
    launcher = (root / "tools/fleet/fleet_launch.sh").read_text()

    assert template["generation"]["native_mcts_hot_loop"] is True
    assert "../../native/gumbel_mcts_rs/Cargo.lock" in template["provenance"][
        "generator_code_files"
    ]
    assert "../../native/gumbel_mcts_rs/src/python_binding.rs" in template[
        "provenance"
    ]["generator_code_files"]
    assert "--native-mcts-hot-loop" in launcher


@pytest.mark.parametrize(
    "relative",
    [
        "configs/guards/generate_gumbel_selfplay_data.json",
        "configs/guards/a1_generation_n128.json",
        "configs/guards/a1_generation_n256.json",
    ],
)
def test_future_generation_guards_bind_native_hot_loop(relative: str) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / relative).read_text())
    lint = next(spec for spec in payload["guards"] if spec["name"] == "cli_flag_lint")

    assert "--native-mcts-hot-loop" in lint["args"]["critical_flags"]
    assert lint["args"]["expected_values"]["--native-mcts-hot-loop"] is True


def test_recomputed_contract_cannot_reuse_real_historical_source_hash(
    tmp_path: Path,
) -> None:
    canonical = contract.GENERATION_CAMPAIGN_CONTRACT_PATH
    payload = json.loads(canonical.read_text())
    # Keep the real, git-recoverable old generator hash but forge a new
    # contract identity and recompute its otherwise-valid semantic digest.
    payload["contract_id"] = "forged-history-reuse"
    payload.pop("contract_sha256")
    payload["contract_sha256"] = contract._digest_value(payload)
    forged = tmp_path / "forged-contract.json"
    forged.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(contract.ContractError, match="immutable file drift"):
        contract.validate_generation_campaign(forged)

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)
from tools.rnd_latent_upgrade_checkpoint import (  # noqa: E402
    SCHEMA_VERSION,
    main,
    sha256_file,
    upgrade_checkpoint,
)


def _checkpoint(
    path: Path,
    *,
    trunk: str = "rrt",
    steps: int = 0,
) -> EntityGraphPolicy:
    config = EntityGraphConfig(
        action_size=16,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=3,
        attention_heads=2,
        dropout=0.0,
        state_trunk=trunk,
        relational_ff_size=24,
        relational_action_cross_layers=1,
        latent_deliberation_steps=steps,
    )
    policy = EntityGraphPolicy(
        config,
        np.zeros((16, LEGAL_ACTION_FEATURE_SIZE), dtype=np.float32),
        seed=3,
        device="cpu",
    )
    policy.save(path, mask_hidden_info=True, soft_target_source="policy")
    return policy


def test_upgrade_preserves_shared_tensors_and_proves_exact_function(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "think-k4.pt"
    incumbent = _checkpoint(source)

    report = upgrade_checkpoint(source, output, steps=4, slots=6)
    upgraded = EntityGraphPolicy.load(output, device="cpu")
    raw = torch.load(output, map_location="cpu", weights_only=False)
    provenance = raw["latent_deliberation_upgrade"]

    assert report["schema_version"] == SCHEMA_VERSION
    assert upgraded.config.state_trunk == "rrt"
    assert upgraded.config.latent_deliberation_steps == 4
    assert upgraded.config.latent_deliberation_slots == 6
    assert upgraded.trained_with_masked_hidden_info
    assert report["output_parameter_count"] > report["source_parameter_count"]
    for name, tensor in incumbent.model.state_dict().items():
        assert torch.equal(upgraded.model.state_dict()[name], tensor), name
    added = set(upgraded.model.state_dict()) - set(incumbent.model.state_dict())
    assert added == set(provenance["added_deliberation_tensors"])
    assert added and all(name.startswith("deliberation_") for name in added)
    assert torch.count_nonzero(upgraded.model.deliberation_fusion.weight) == 0
    assert torch.count_nonzero(upgraded.model.deliberation_fusion.bias) == 0

    verification = provenance["function_preserving_verification"]
    assert verification["exact"] is True
    assert set(verification["verified_outputs"]) >= {
        "logits",
        "value",
        "final_vp",
        "q_values",
    }
    assert provenance["source_checkpoint_sha256"] == sha256_file(source)
    assert report["output_checkpoint_sha256"] == sha256_file(output)
    for relative, digest in provenance["implementation_sha256"].items():
        assert digest == hashlib.sha256(Path(relative).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "trunk,steps,error",
    [
        ("transformer", 0, "state_trunk='rrt'"),
        ("rrt", 1, "must be a K=0"),
    ],
)
def test_upgrade_rejects_non_k0_rrt_sources(tmp_path, trunk, steps, error):
    source = tmp_path / "source.pt"
    _checkpoint(source, trunk=trunk, steps=steps)
    with pytest.raises(ValueError, match=error):
        upgrade_checkpoint(source, tmp_path / "output.pt", steps=2, slots=8)


@pytest.mark.parametrize(
    "steps,slots,error",
    [(0, 8, "steps must be >= 1"), (2, 0, "slots must be >= 1")],
)
def test_upgrade_rejects_invalid_requested_shape(tmp_path, steps, slots, error):
    source = tmp_path / "source.pt"
    _checkpoint(source)
    with pytest.raises(ValueError, match=error):
        upgrade_checkpoint(source, tmp_path / "output.pt", steps=steps, slots=slots)


def test_upgrade_is_atomic_and_never_overwrites(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "think.pt"
    _checkpoint(source)
    upgrade_checkpoint(source, output, steps=2, slots=8)
    original = output.read_bytes()

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        upgrade_checkpoint(source, output, steps=4, slots=8)
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".think.pt.tmp.*"))
    with pytest.raises(ValueError, match="must differ"):
        upgrade_checkpoint(source, source, steps=2, slots=8)


def test_cli_writes_no_overwrite_json_report(tmp_path, capsys):
    source = tmp_path / "source.pt"
    output = tmp_path / "think.pt"
    report = tmp_path / "report.json"
    _checkpoint(source)
    assert main(
        [
            "--source",
            str(source),
            "--output",
            str(output),
            "--steps",
            "2",
            "--slots",
            "5",
            "--report",
            str(report),
        ]
    ) == 0
    assert '"output_checkpoint_sha256"' in report.read_text(encoding="utf-8")
    assert capsys.readouterr().out == report.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        main(
            [
                "--source",
                str(source),
                "--output",
                str(tmp_path / "other.pt"),
                "--steps",
                "2",
                "--report",
                str(report),
            ]
        )

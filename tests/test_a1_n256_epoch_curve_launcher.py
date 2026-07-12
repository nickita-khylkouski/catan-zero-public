from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools/a1_n256_epoch_curve_b200.sh"


def test_epoch_curve_launcher_is_clean_producer_started_and_nonpromotable() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'overrides=$(printf \'{"epochs":3,"loser_sample_weight":1.0,"lr":%s}\'' in text
    assert 'assert "--no-resume-optimizer" in cmd' in text
    assert '"--save-each-epoch" in cmd' in text
    assert '"--train-diagnostics-every-batches"' in text
    assert '== "100"' in text
    assert 'set(a["recipe_drift"]) == {"epochs","lr","loser_sample_weight"}' in text
    assert '"exposure_points":[1.0,2.0,3.0]' in text
    assert '"half_epoch_checkpoint":None' in text
    assert "only validated epoch-boundary checkpoints are comparable" in text
    assert '"promotion_eligible":False' in text
    assert '"launch_authorized":False' in text
    assert '"pairs_per_checkpoint":64' in text


def test_epoch_curve_launcher_requires_explicit_anchor_receipt() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert "--anchor-receipt" in result.stderr

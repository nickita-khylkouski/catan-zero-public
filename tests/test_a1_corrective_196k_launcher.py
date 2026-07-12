from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "tools/a1_corrective_196k_b200.sh"


def test_corrective_launcher_is_syntax_clean_and_fail_closed_on_handoff() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "a1_combined_candidate_handoff" in text
    assert "A1_HANDOFF_REPO" in text
    assert '[[ -f "$root/n128.training_input.ready" ]]' in text
    assert "combined.verify_result" in text
    assert 'result["passed"]' in text
    assert "final.summary.json" not in text
    assert "--go" in text
    assert "lr=1.2e-4" in text
    assert 'overrides=\'{"loser_sample_weight":1.0,"lr":0.00012}\'' in text
    assert "--world-size 8" in text
    assert "--curriculum-parent-receipt" in text
    assert "decoder.raw_decode" in text
    assert "unexpected dual learner dry-run stdout stream" in text

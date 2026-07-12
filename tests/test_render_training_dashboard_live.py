import json

from tools import render_training_dashboard as dashboard


def test_live_bc_batch_stream_sets_phase_and_curves(tmp_path, monkeypatch) -> None:
    events = [
        {"progress": "bc_batch", "arch": "entity_graph", "epoch": 1,
         "batch": 50, "batches": 100, "samples": 25_600,
         "loss": 1.5, "accuracy": 0.6},
        {"progress": "bc_batch", "arch": "entity_graph", "epoch": 1,
         "batch": 100, "batches": 100, "samples": 51_200,
         "loss": 1.2, "accuracy": 0.7},
    ]
    (tmp_path / "train.log").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    monkeypatch.setattr(dashboard, "_gpu_snapshot", lambda: {})

    status = dashboard.build_status(tmp_path, pid=0, target_games=0)

    assert status["phase"] == "training_bc"
    assert status["training"]["latest"]["batch"] == 100
    assert status["training"]["curve"] == events
    rendered = dashboard._html(status, refresh_seconds=10)
    assert "Optimizer Progress" in rendered
    assert "100 / 100" in rendered
    assert 'aria-label="loss curve"' in rendered

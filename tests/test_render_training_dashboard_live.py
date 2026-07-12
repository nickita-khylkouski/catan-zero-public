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


def test_json_stream_accepts_pretty_plan_and_adjacent_events(tmp_path) -> None:
    first = {"progress": "bc_memmap_load", "rows": 10}
    plan = {"schema_version": "plan", "nested": {"world_size": 8}}
    batch = {"progress": "bc_batch", "arch": "entity_graph", "batch": 1}
    path = tmp_path / "train.log"
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(plan, indent=2) + json.dumps(batch) + "\n",
        encoding="utf-8",
    )

    assert dashboard._parse_json_stream(path) == [first, plan, batch]


def test_completed_generic_report_wins_over_stale_batch(tmp_path, monkeypatch) -> None:
    (tmp_path / "train.log").write_text(
        json.dumps({"progress": "bc_batch", "arch": "entity_graph", "batch": 1}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "report.json").write_text(
        json.dumps({"arch": "entity_graph", "metrics": [{"epoch": 1, "loss": 1.0}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "_gpu_snapshot", lambda: {})

    assert dashboard.build_status(tmp_path, pid=0, target_games=0)["phase"] == "complete"

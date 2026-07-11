from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / "ops" / "observability"


def _load_generator():
    path = OBS / "gen_targets.py"
    spec = importlib.util.spec_from_file_location("catan_obs_gen_targets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_renderer_adds_node_dcgm_and_catan_without_gpu_label(
    tmp_path: Path,
) -> None:
    generator = _load_generator()
    config = {
        "hub": {
            "name": "b200",
            "cluster": "b200",
            "role": "rnd",
            "gpumodel": "B200",
        },
        "boxes": [
            {
                "name": "c1",
                "host": "192.0.2.1",
                "cluster": "c1",
                "role": "generation",
                "gpumodel": "H100",
            }
        ],
    }
    result = generator.render(
        config, out_dir=tmp_path / "generated", ssh_key=Path("/private/key")
    )
    assert result == {"boxes": 2, "tunnels": 1}
    for job in ("node", "dcgm"):
        records = json.loads(
            (tmp_path / "generated/targets" / f"{job}.json").read_text()
        )
        assert len(records) == 2
        assert all("gpu" not in record["labels"] for record in records)
        assert all("gpumodel" in record["labels"] for record in records)
    catan = json.loads(
        (tmp_path / "generated/targets/catan.json").read_text()
    )
    assert len(catan) == 1
    assert catan[0]["targets"] == ["localhost:19501"]
    assert "gpu" not in catan[0]["labels"]
    assert catan[0]["labels"]["gpumodel"] == "H100"
    unit = (tmp_path / "generated/tunnels/fleet-tunnel-c1.service").read_text()
    assert "127.0.0.1:19101:127.0.0.1:9100" in unit
    assert "127.0.0.1:19401:127.0.0.1:9400" in unit
    assert "127.0.0.1:19501:127.0.0.1:9500" in unit


def test_target_renderer_rejects_systemd_injection(tmp_path: Path) -> None:
    generator = _load_generator()
    config = {
        "hub": {
            "name": "b200",
            "cluster": "b200",
            "role": "rnd",
            "gpumodel": "B200",
        },
        "boxes": [
            {
                "name": "c1\nExecStart=/bin/false",
                "host": "192.0.2.1",
                "cluster": "c1",
                "role": "generation",
                "gpumodel": "H100",
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="safe label"):
        generator.render(
            config,
            out_dir=tmp_path / "generated",
            ssh_key=Path("/private/key"),
        )


def test_target_renderer_prunes_removed_box_units(tmp_path: Path) -> None:
    generator = _load_generator()
    config = {
        "hub": {
            "name": "b200",
            "cluster": "b200",
            "role": "rnd",
            "gpumodel": "B200",
        },
        "boxes": [
            {
                "name": name,
                "host": f"192.0.2.{index}",
                "cluster": name,
                "role": "generation",
                "gpumodel": "H100",
            }
            for index, name in enumerate(("c1", "c2"), start=1)
        ],
    }
    out = tmp_path / "generated"
    generator.render(config, out_dir=out, ssh_key=Path("/private/key"))
    assert (out / "tunnels/fleet-tunnel-c2.service").exists()

    config["boxes"] = config["boxes"][:1]
    generator.render(config, out_dir=out, ssh_key=Path("/private/key"))
    assert not (out / "tunnels/fleet-tunnel-c2.service").exists()


def test_committed_dashboard_covers_required_gpu_and_generator_metrics() -> None:
    dashboard_path = OBS / "grafana/dashboards/catan_fleet_production.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert dashboard["uid"] == "catan-zero-production"
    expressions = "\n".join(
        target.get("expr", "")
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    )
    for metric in (
        "DCGM_FI_DEV_GPU_UTIL",
        "DCGM_FI_DEV_FB_USED",
        "DCGM_FI_DEV_POWER_USAGE",
        "DCGM_FI_DEV_GPU_TEMP",
        "catan_fleet_generator_processes",
        "catan_fleet_generator_healthy",
        "catan_fleet_generator_games_completed",
        "catan_fleet_generator_rows",
        "catan_fleet_generator_simulations",
        "catan_fleet_generator_shards",
        "catan_fleet_generator_failures",
        "catan_fleet_generator_truncations",
        "catan_fleet_generator_progress_age_seconds",
        "catan_fleet_generator_info",
        "catan_fleet_output_disk_free_bytes",
    ):
        assert metric in expressions
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert "Active A1 lanes / expected 56" in titles
    assert "Recipe-safe active lanes / expected 56" in titles
    arm_panel = next(
        panel
        for panel in dashboard["panels"]
        if panel["title"] == "Active lanes by search budget / expected 28 each"
    )
    arm_query = arm_panel["targets"][0]["expr"]
    assert 'count_values("n_full", catan_fleet_generator_n_full' in arm_query
    assert "catan_fleet_generator_processes" in arm_query
    assert "> 0" in arm_query
    assert arm_panel["fieldConfig"]["defaults"]["max"] == 28
    for title in (
        "Active A1 lanes / expected 56",
        "Recipe-safe active lanes / expected 56",
    ):
        panel = next(panel for panel in dashboard["panels"] if panel["title"] == title)
        assert panel["fieldConfig"]["defaults"]["max"] == 56


def test_dashboard_box_filter_uses_prometheus_fleet_alias_consistently() -> None:
    dashboard_path = OBS / "grafana/dashboards/catan_fleet_production.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    variables = dashboard["templating"]["list"]
    assert len(variables) == 1
    box = variables[0]
    assert box["name"] == "box"
    assert box["query"]["query"] == "label_values(catan_fleet_exporter_up, box)"
    assert box["includeAll"] is True
    assert box["current"] == {"text": "All", "value": "$__all"}

    catan_targets = [
        target
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
        if "catan_fleet_" in target.get("expr", "")
    ]
    assert catan_targets
    for target in catan_targets:
        expression = target["expr"]
        assert 'box=~"$box"' in expression
        assert 'host=~"$box"' not in expression

    rate_panel = next(
        panel
        for panel in dashboard["panels"]
        if panel["title"] == "Rows and simulations rate"
    )
    assert all("sum by(box)" in target["expr"] for target in rate_panel["targets"])
    assert {target["legendFormat"] for target in rate_panel["targets"]} == {
        "{{box}} rows/s",
        "{{box}} sims/s",
    }


def test_prometheus_and_grafana_provisioning_are_committed() -> None:
    prometheus = (OBS / "prometheus/prometheus.yml").read_text(encoding="utf-8")
    assert "job_name: dcgm" in prometheus
    assert "job_name: node" in prometheus
    assert "job_name: catan-generator" in prometheus
    assert "/etc/prometheus/targets/catan.json" in prometheus
    datasource = (
        OBS / "grafana/provisioning/datasources/prometheus.yml"
    ).read_text(encoding="utf-8")
    dashboards = (
        OBS / "grafana/provisioning/dashboards/dashboards.yml"
    ).read_text(encoding="utf-8")
    assert "uid: prometheus" in datasource
    assert "/var/lib/grafana/dashboards" in dashboards
    service = (OBS / "systemd/catan-fleet-exporter.service").read_text()
    assert "--listen 127.0.0.1 --port 9500" in service
    assert "/home/ubuntu/catan-zero-v1/.venv/bin/python" in service
    assert "--run-root /home/ubuntu/gen_out" in service
    assert "--run-root /home/ubuntu/catan-zero-production/runs/selfplay" in service
    assert "NoNewPrivileges=true" in service
    compose = (OBS / "docker-compose.yml").read_text()
    assert "GF_SERVER_HTTP_ADDR=127.0.0.1" in compose
    rules = (OBS / "prometheus/rules/catan_fleet.yml").read_text()
    assert "CatanGeneratorExitedIncomplete" in rules
    assert "catan_fleet_generator_processes == 0" in rules

# Production fleet observability

This directory source-controls the existing B200-hub architecture instead of
creating a second monitoring stack:

- DCGM exporter on each GPU box: per-GPU utilization, FB memory, power, and
  temperature on loopback port `9400`.
- node-exporter: host CPU/memory/filesystem on loopback port `9100`.
- `tools/fleet/fleet_metrics_exporter.py`: Catan generator/process/progress
  metrics on loopback port `9500`.
- SSH systemd tunnels terminate on the hub; Prometheus and Grafana run there.

Nothing in this directory deploys itself. Install only after the release/tree,
fleet target file, SSH key, and one-host canary have been reviewed.

## Per-box canary

The exporter is read-only. It scans `/proc`, `~/gen_out/*/gpu*/config.json`,
worker `progress.json` files, final `manifest.json` files, and filesystem free
space. The fleet launcher writes the existing typed config format to
`$GPU_OUT/config.json` before workers start, making the canonical config hash
available while a run is live.

```bash
python tools/fleet/fleet_metrics_exporter.py --once --run-root ~/gen_out
python tools/fleet/fleet_metrics_exporter.py --listen 127.0.0.1 --port 9500
curl -fsS http://127.0.0.1:9500/metrics | grep '^catan_fleet_'
```

After the one-shot output is correct, install the pinned unit from
`systemd/catan-fleet-exporter.service`. It follows the canonical fleet launcher
defaults (`/home/ubuntu/catan-zero-v1` and `/home/ubuntu/gen_out`); change those
paths only in the same reviewed change as the launcher. The unit binds loopback,
runs unprivileged, and has a read-only home/system view.

## Hub provisioning

Copy `fleet_targets.example.json` to a private file and replace every host.
Never commit the private target file.

```bash
python ops/observability/gen_targets.py \
  --config /private/fleet_targets.json \
  --out-dir ops/observability/generated \
  --ssh-key /home/ubuntu/.ssh/gpu_access_ed25519
```

Review generated target JSON and tunnel units. Each remote box uses three
localhost forwards: node `191xx`, DCGM `194xx`, Catan `195xx`. Install units on
the hub only after `ssh -N` canaries succeed. Then create a mode-600 `.env`
containing `GF_ADMIN_PW=...` and run:

```bash
cd ops/observability
docker compose config
docker compose up -d
curl -fsS -X POST http://127.0.0.1:9090/-/reload
```

Prometheus is loopback-only. Grafana provisions the `Catan-Zero Production
Fleet` dashboard and its Prometheus datasource from committed files. Prefer an
SSH tunnel for Grafana access; if port 3000 is public, firewall it and add TLS.

## Metric contract

DCGM supplies per-host/per-GPU utilization, memory, power, and temperature.
The Catan exporter supplies, per current host/GPU/pipeline/run/config:

- `generator_processes`, `generator_healthy`, `generator_complete`;
- `generator_games_requested/completed`, `generator_rows`,
  `generator_simulations`, `generator_shards`;
- `generator_failures`, `generator_truncations`;
- `generator_progress_age_seconds`;
- `generator_info` labels with the typed config hash, seed range, target
  information regime, and every exact A1 safety-critical recipe field
  (public observation, information-set search, determinization particle
  schedule, n_full/n_fast/p_full, symmetry settings, c_scale/c_visit,
  max_depth, lazy chance, and legacy belief chance);
- numeric per-lane recipe gauges, `generator_recipe_safe`, target-regime
  attestation state, and per-host active/recipe-safe lane totals;
- output filesystem free/total bytes.

Only the newest run per GPU/pipeline slot is exported, preferring an active
process, which bounds Prometheus label cardinality to the launcher's maximum of
two pipelines per GPU. Completed runs remain healthy; an active
run is unhealthy after five minutes without an atomic progress update or after
any worker failure. The dashboard displays truncations separately because a
bounded, declared truncation is not the same as a worker crash.

## Alerts and rollout checks

Committed rules cover exporter/tunnel loss, stale or exited-incomplete
generation, worker failures, exact-recipe mismatch, low output disk, DCGM
loss, and an active-but-idle H100 lane. The GPU-idle rule is joined to an
active Catan lane and explicitly scoped to `gpumodel="H100"`; B200 R&D/training
activity cannot trigger it. Before expanding beyond one host:

1. confirm all three targets are `UP`;
2. compare exporter games/rows/sims/shards against the underlying JSON;
3. confirm DCGM GPU labels stay device indices (`gpumodel` is the target label);
4. stop a canary exporter and verify `CatanExporterDown` becomes pending;
5. use stale and exited-incomplete fixtures and verify both generator alerts;
6. deliberately alter one canary recipe field and verify
   `CatanGeneratorRecipeMismatch` fires;
7. verify the dashboard reaches exactly 56 active and 56 recipe-safe lanes;
8. verify no exporter port listens on a public interface.

The dashboard is observability only. It never authorizes harvest, training,
promotion, stopping, or seed reuse.

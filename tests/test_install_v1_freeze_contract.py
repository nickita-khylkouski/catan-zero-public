from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "tools" / "install_v1_freeze.sh"
MPS_UNIT = ROOT / "tools" / "fleet" / "systemd" / "nvidia-mps.service"
WHEEL_NAME = "catanatron_rs-0.1.8-cp311-cp311-manylinux_2_34_x86_64.whl"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _release_repo(tmp_path: Path, inventory: str) -> Path:
    repo = tmp_path / "release-repo"
    (repo / "native" / "catanatron-rs").mkdir(parents=True)
    (repo / "tools" / "fleet" / "systemd").mkdir(parents=True)
    (repo / "native" / "catanatron-rs" / "WHEEL_SHA256SUMS").write_text(
        inventory, encoding="utf-8"
    )
    (repo / "tools" / "fleet" / "systemd" / "nvidia-mps.service").write_text(
        "[Service]\nType=simple\nExecStart=/bin/true\n", encoding="utf-8"
    )
    (repo / ".gitignore").write_text(".venv/\n*.ignored\n", encoding="utf-8")
    _run("git", "init", "-q", "-b", "main", cwd=repo)
    _run("git", "config", "user.email", "installer-test@example.invalid", cwd=repo)
    _run("git", "config", "user.name", "Installer Test", cwd=repo)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-q", "-m", "release fixture", cwd=repo)
    _run("git", "tag", "vtest", cwd=repo)
    return repo


def _fake_sudo(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "sudo-was-called"
    sudo = fake_bin / "sudo"
    sudo.write_text(
        '#!/usr/bin/env bash\nset -eu\n: > "$SUDO_MARKER"\nexit 99\n',
        encoding="utf-8",
    )
    sudo.chmod(0o755)
    return fake_bin, marker


def _invoke(
    tmp_path: Path,
    *,
    repo: Path,
    wheel: Path,
    destination: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    fake_bin, marker = _fake_sudo(tmp_path)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    destination = destination or (tmp_path / "deployment")
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "SUDO_MARKER": str(marker),
        "CATAN_REPO": str(repo),
        "CATAN_REF": "vtest",
        "CATAN_DEST": str(destination),
        "CATAN_RS_WHEEL": str(wheel),
        "PY": "python3.11-does-not-run-before-wheel-preflight",
    }
    result = subprocess.run(
        ["bash", str(INSTALLER)],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result, marker, destination


def _wheel(tmp_path: Path, *, name: str = WHEEL_NAME) -> Path:
    path = tmp_path / name
    path.write_bytes(b"test wheel bytes\n")
    return path


def test_installer_shell_syntax_and_preflight_order() -> None:
    subprocess.run(["bash", "-n", str(INSTALLER)], check=True)
    text = INSTALLER.read_text(encoding="utf-8")
    verified = text.index("catanatron_rs wheel preflight verified")
    mutations = (
        text.index('sudo install -m 0644 "$MPS_UNIT_SOURCE"'),
        text.index('"$PY" -m venv .venv'),
        text.index('python -m pip install "torch>=2.11"'),
    )
    assert all(verified < mutation for mutation in mutations)
    assert 'git rev-parse --verify "refs/tags/${CATAN_REF}^{commit}"' in text
    assert "sha256sum -c --strict wheel.sha256" in text
    assert "CATAN_DEST must be an absolute systemd-safe path" in text
    assert 'assert rs == "0.1.8"' in text
    assert 'getattr(catanatron_rs, "gumbel_search", None)' in text
    assert 'rust_version != "0.1.8"' in text
    assert "gumbel_search_capabilities" in text
    assert "sigma_reference_visits" in text
    assert "belief_target_evidence" in text
    assert "initial_road_d1_scope" in text
    assert "public_award_feature_parity" in text


def test_malformed_inventory_fails_before_any_privileged_mutation(
    tmp_path: Path,
) -> None:
    wheel = _wheel(tmp_path)
    repo = _release_repo(tmp_path, f"not-a-sha256  {WHEEL_NAME}\n")
    result, sudo_marker, destination = _invoke(tmp_path, repo=repo, wheel=wheel)
    assert result.returncode == 3
    assert "malformed Rust-wheel checksum inventory" in result.stdout
    assert not sudo_marker.exists()
    assert not (destination / ".venv").exists()


def test_duplicate_inventory_entry_fails_closed(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    repo = _release_repo(
        tmp_path,
        f"{digest}  {WHEEL_NAME}\n{digest}  {WHEEL_NAME}\n",
    )
    result, sudo_marker, _destination = _invoke(tmp_path, repo=repo, wheel=wheel)
    assert result.returncode == 3
    assert "must contain exactly one non-empty record" in result.stdout
    assert not sudo_marker.exists()


def test_additional_well_formed_inventory_record_fails_closed(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    repo = _release_repo(
        tmp_path,
        f"{digest}  {WHEEL_NAME}\n"
        f"{'1' * 64}  catanatron_rs-0.1.8-cp310-cp310-manylinux_2_34_x86_64.whl\n",
    )
    result, sudo_marker, _destination = _invoke(tmp_path, repo=repo, wheel=wheel)
    assert result.returncode == 3
    assert "must contain exactly one non-empty record" in result.stdout
    assert "records=2 matches=1" in result.stdout
    assert not sudo_marker.exists()


def test_wrong_wheel_digest_fails_before_any_privileged_mutation(
    tmp_path: Path,
) -> None:
    wheel = _wheel(tmp_path)
    repo = _release_repo(tmp_path, f"{'0' * 64}  {WHEEL_NAME}\n")
    result, sudo_marker, destination = _invoke(tmp_path, repo=repo, wheel=wheel)
    assert result.returncode == 3
    assert "digest mismatch" in result.stdout
    assert not sudo_marker.exists()
    assert not (destination / ".venv").exists()


def test_wrong_wheel_filename_fails_closed(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, name="renamed.whl")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    repo = _release_repo(tmp_path, f"{digest}  {WHEEL_NAME}\n")
    result, sudo_marker, _destination = _invoke(tmp_path, repo=repo, wheel=wheel)
    assert result.returncode == 3
    assert "wrong filename" in result.stdout
    assert not sudo_marker.exists()


def test_valid_wheel_is_verified_before_sudo_is_reached(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    repo = _release_repo(tmp_path, f"{digest}  {WHEEL_NAME}\n")
    result, sudo_marker, destination = _invoke(tmp_path, repo=repo, wheel=wheel)
    assert result.returncode == 3
    assert f"wheel preflight verified: {digest}" in result.stdout
    assert "passwordless sudo is required" in result.stdout
    assert sudo_marker.exists()
    assert not (destination / ".venv").exists()


def test_dirty_or_reused_destination_is_rejected_before_sudo(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    repo = _release_repo(tmp_path, f"{digest}  {WHEEL_NAME}\n")
    destination = tmp_path / "existing-deployment"
    _run("git", "clone", "-q", str(repo), str(destination), cwd=tmp_path)
    (destination / "operator-drift.txt").write_text("drift\n", encoding="utf-8")
    result, sudo_marker, _destination = _invoke(
        tmp_path, repo=repo, wheel=wheel, destination=destination
    )
    assert result.returncode == 3
    assert "deployment checkout is dirty" in result.stdout
    assert not sudo_marker.exists()

    (destination / "operator-drift.txt").unlink()
    (destination / "shadow.ignored").write_text("ignored drift\n", encoding="utf-8")
    ignored_root = tmp_path / "ignored"
    ignored_root.mkdir()
    result, sudo_marker, _destination = _invoke(
        ignored_root, repo=repo, wheel=wheel, destination=destination
    )
    assert result.returncode == 3
    assert "deployment checkout is dirty" in result.stdout
    assert not sudo_marker.exists()

    (destination / "shadow.ignored").unlink()
    (destination / ".venv").mkdir()
    # _invoke creates its fake-bin under tmp_path; use a separate test root for
    # the second independent invocation.
    second_root = tmp_path / "second"
    second_root.mkdir()
    result, sudo_marker, _destination = _invoke(
        second_root, repo=repo, wheel=wheel, destination=destination
    )
    assert result.returncode == 3
    assert "already contains .venv" in result.stdout
    assert not sudo_marker.exists()


def test_receipt_is_outside_checkout_and_binds_release_evidence() -> None:
    text = INSTALLER.read_text(encoding="utf-8")
    assert "deployment checkout drifted during installation" in text
    assert text.index("deployment checkout drifted during installation") < text.index(
        '"schema_version": "catan-zero-install-receipt-v2"'
    )
    assert '"schema_version": "catan-zero-install-receipt-v2"' in text
    assert "$HOME/.local/state/catan-zero/install-${HEAD_COMMIT}.json" in text
    for field in (
        '"source_commit"',
        '"tag_commit"',
        '"sha256"',
        '"expected_sha256"',
        '"checksum_inventory_sha256"',
        '"catanatron_rs_version"',
        '"determinize_for_player"',
        '"gumbel_search"',
        '"fleet_exporter_fragment_path"',
        '"fleet_exporter_dropin_paths"',
        '"fleet_exporter_effective"',
        '"nvidia_mps_limit_nofile_soft"',
    ):
        assert field in text
    assert "handle.flush()" in text
    assert "os.fsync(handle.fileno())" in text
    assert "os.replace(temporary, receipt)" in text


def test_mps_unit_and_installer_bind_effective_nofile_limit() -> None:
    unit = MPS_UNIT.read_text(encoding="utf-8")
    assert unit.splitlines().count("LimitNOFILE=65536") == 1

    text = INSTALLER.read_text(encoding="utf-8")
    restart = text.index("sudo systemctl restart nvidia-mps.service")
    inspect = text.index(
        "systemctl show nvidia-mps.service --property=LimitNOFILESoft --value"
    )
    receipt = text.index('"nvidia_mps_limit_nofile_soft"')
    assert restart < inspect < receipt
    assert "MPS_REQUIRED_LIMIT_NOFILE_SOFT=65536" in text
    assert '[[ ! "$CATAN_MPS_LIMIT_NOFILE_SOFT" =~ ^[0-9]+$ ]]' in text
    assert '"$CATAN_MPS_LIMIT_NOFILE_SOFT" -lt "$MPS_REQUIRED_LIMIT_NOFILE_SOFT"' in text
    assert "export CATAN_MPS_LIMIT_NOFILE_SOFT" in text


def test_exporter_install_removes_legacy_dropins_and_attests_effective_process() -> None:
    text = INSTALLER.read_text(encoding="utf-8")
    assert 'EXPORTER_DROPIN_DIR="/etc/systemd/system/catan-fleet-exporter.service.d"' in text
    assert 'sudo rm -rf -- "$EXPORTER_DROPIN_DIR"' in text
    assert "--property=FragmentPath" in text
    assert "--property=DropInPaths" in text
    assert 'Path(f"/proc/{pid}/cmdline")' in text
    assert '"--run-root", "/home/ubuntu/gen_out"' in text
    assert '"--run-root", "/home/ubuntu/catan-zero-production/runs/selfplay"' in text
    assert "build_opener(ProxyHandler({}))" in text
    assert '"catan_fleet_" not in body' in text
    assert "MainPID changed during attestation" in text
    assert "finish_install_transaction" in text
    assert "EXPORTER_TRANSACTION_ARMED=1" in text
    assert "EXPORTER_TRANSACTION_ARMED=0" in text
    assert "disable --now catan-fleet-exporter.service" in text
    assert "exporter rollback FAILED" in text
    assert 'rm -f -- "$CATAN_INSTALL_RECEIPT"' in text
    assert "--property=ActiveState" in text
    assert "--property=UnitFileState" in text
    assert "--property=MainPID" in text
    assert '[ "$active" != "inactive" ]' in text
    assert '[ "$enabled" != "disabled" ]' in text
    assert '[ "$main_pid" != "0" ]' in text
    assert 'rm -f -- "$CATAN_INSTALL_RECEIPT"' in text

    remove = text.index('sudo rm -rf -- "$EXPORTER_DROPIN_DIR"')
    install = text.index('sudo install -m 0644 "$EXPORTER_UNIT_RENDERED"')
    receipt = text.index('"schema_version": "catan-zero-install-receipt-v2"')
    assert remove < install < receipt


def test_exporter_unit_is_rendered_for_a_safe_fresh_destination() -> None:
    text = INSTALLER.read_text(encoding="utf-8")
    assert 'default = "/home/ubuntu/catan-zero-v1"' in text
    assert "text.count(default) != 2" in text
    assert "rendered = text.replace(default, str(destination))" in text
    assert 're.fullmatch(r"/[A-Za-z0-9._/-]+", str(destination))' in text

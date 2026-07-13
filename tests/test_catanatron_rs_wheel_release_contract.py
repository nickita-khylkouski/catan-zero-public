"""Release-contract tests for the sealed ``catanatron_rs`` wheel builder.

These tests deliberately inspect the small shell entry point rather than build
the Rust extension.  A native release build is a B200 acceptance test; CI's job
is to prevent changes that silently weaken the inputs to that build.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "tools" / "build_catanatron_rs_wheel.sh"
CHECKSUM_INVENTORY = "native/catanatron-rs/WHEEL_SHA256SUMS"
RECEIPT_NAME = "catanatron_rs-0.1.8-build-receipt.json"
RECEIPT_SCHEMA = "catanatron-rs-wheel-build-receipt-v2"


def _script() -> str:
    return BUILDER.read_text()


def _flat_script() -> str:
    """Collapse shell line continuations/whitespace for command assertions."""

    return " ".join(_script().replace("\\\n", " ").split())


def test_builder_is_valid_bash_and_stages_the_committed_tree_at_one_root() -> None:
    subprocess.run(["bash", "-n", str(BUILDER)], check=True)
    script = _script()

    assert "/tmp/catan-zero-catanatron-rs-wheel-src" in script
    assert re.search(r"git\s+-C\s+.*\sarchive\s+", _flat_script())
    assert "CATAN_RS_BUILD_STAGED" in script
    assert 'SOURCE_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse --verify HEAD)"' in script
    assert "build_receipt_final" in script
    assert "canonical_build_root" in script


def test_builder_guards_the_recursive_delete_root_before_staging() -> None:
    script = _script()
    rm_offset = script.index('rm -rf "$CANONICAL_BUILD_ROOT"')
    prefix = script[:rm_offset]

    # The guard must be an explicit release contract, not reliance on rm's
    # implementation-specific protection for '/'.  It also protects the
    # checkout and an empty/unresolved path from a bad environment override.
    assert '"$CANONICAL_BUILD_ROOT" = "$SEALED_CANONICAL_BUILD_ROOT"' in prefix
    assert '"$CANONICAL_BUILD_ROOT" != "$SOURCE_ROOT"' in prefix
    assert '"$CANONICAL_BUILD_ROOT" != "$OUT_DIR"' in prefix
    assert "must be an absolute, dedicated" in prefix


def test_checksum_inventory_is_not_an_input_to_the_wheel() -> None:
    script = _script()
    staged_call = script.index(
        '"$CANONICAL_BUILD_ROOT/tools/build_catanatron_rs_wheel.sh"'
    )
    prefix = script[:staged_call]

    # The checksum belongs to commit B of the release transaction.  Including
    # it in commit A's staged source makes the digest inventory affect the
    # binary it is meant to describe, creating a circular/non-reproducible
    # release input.
    assert CHECKSUM_INVENTORY in prefix
    assert re.search(r"(?:rm\s+-f|--exclude(?:=|\s+))[^\n]*WHEEL_SHA256SUMS", prefix)
    assert "checksum_inventory_excluded" in script


def test_builder_rejects_dirty_tracked_or_staged_changes() -> None:
    script = _flat_script()

    assert re.search(r"git -C .* diff --quiet --exit-code", script)
    assert re.search(r"git -C .* diff --cached --quiet --exit-code", script)
    assert "refusing to build a release wheel from a dirty tracked tree" in script


def test_all_cargo_resolution_is_locked() -> None:
    script = _flat_script()

    assert "cargo test --locked" in script
    assert "maturin build --locked" in script
    assert "native/catanatron-rs/Cargo.lock" in _script()
    assert "native/catanatron-rs/python/Cargo.lock" in _script()
    assert "native/gumbel_mcts_rs/Cargo.lock" in _script()


def test_native_mcts_wheel_has_one_unique_018_package_identity() -> None:
    expected = "0.1.8"
    manifests = (
        REPO_ROOT / "native/catanatron-rs/Cargo.toml",
        REPO_ROOT / "native/catanatron-rs/python/Cargo.toml",
        REPO_ROOT / "native/catanatron-rs/pyproject.toml",
    )
    for manifest in manifests:
        match = re.search(r'^version = "([^"]+)"$', manifest.read_text(), re.MULTILINE)
        assert match is not None
        assert match.group(1) == expected

    script = _script()
    assert "catanatron_rs-0.1.8-cp311-cp311-manylinux_2_34_x86_64.whl" in script
    assert "catanatron_rs-0.1.8-build-receipt.json" in script
    assert "catanatron_rs-0.1.4-cp311" not in script


def test_release_environment_and_toolchain_are_sealed() -> None:
    script = _script()

    assert "env -i" in script
    assert "1784073600" in script
    assert "--remap-path-prefix=" in script
    assert "-C link-arg=-Wl,--build-id=none" in script
    assert re.search(r"CARGO_BUILD_JOBS=(?:['\"]?1['\"]?)", script)

    # These are exact release inputs.  Merely printing whatever happens to be
    # installed is insufficient: a mismatched tool must fail before building.
    for version in (
        "rustc 1.96.1 (31fca3adb 2026-06-26)",
        "cargo 1.96.1 (356927216 2026-06-26)",
        "maturin 1.14.1",
        "Python 3.11.15",
        "GNU strip (GNU Binutils for Ubuntu) 2.38",
    ):
        assert version in script


def test_source_identity_is_bound_only_after_compilation() -> None:
    script = _script()
    staged_environment = script[
        script.index("env -i") : script.index(
            '"$CANONICAL_BUILD_ROOT/tools/build_catanatron_rs_wheel.sh"'
        )
    ]
    assert 'CATAN_RS_SOURCE_COMMIT="$SEALED_COMPILE_IDENTITY"' in staged_environment
    assert 'CATAN_RS_SOURCE_TREE="$SEALED_COMPILE_IDENTITY"' in staged_environment
    assert 'CATAN_RS_SOURCE_COMMIT="$SOURCE_COMMIT"' not in staged_environment
    assert 'CATAN_RS_SOURCE_TREE="$SOURCE_TREE"' not in staged_environment
    assert 'SEALED_COMPILE_IDENTITY="catanatron-rs-0.1.8-public-award-temperature-wheel-v1"' in script
    assert 'payload["source_commit"] = sys.argv[2]' in script
    assert 'payload["source_tree"] = sys.argv[3]' in script
    assert '"source_commit": None' in script
    assert '"source_tree": None' in script


def test_builder_emits_a_complete_machine_readable_receipt() -> None:
    script = _script()

    assert RECEIPT_NAME in script
    assert RECEIPT_SCHEMA in script
    for key in (
        "source_commit",
        "source_tree",
        "builder_sha256",
        "cargo_lock_sha256",
        "python_cargo_lock_sha256",
        "gumbel_cargo_lock_sha256",
        "gumbel_lib_rs_sha256",
        "gumbel_python_binding_rs_sha256",
        "rustc_version",
        "cargo_version",
        "maturin_version",
        "python_version",
        "strip_version",
        "elf_normalization",
        "canonical_build_root",
        "compile_identity",
        "source_date_epoch",
        "rustflags",
        "cargo_build_jobs",
        "checksum_inventory_excluded",
        "wheel_filename",
        "wheel_sha256",
    ):
        assert f'"{key}"' in script or f"'{key}'" in script


def test_builder_smokes_the_compiled_capability_contract_before_hashing() -> None:
    script = _script()
    smoke = script.index('PYTHONPATH="$NORMALIZE_TMP"')
    digest = script.index('WHEEL_SHA256="$(sha256sum "$WHEEL_PATH"')

    assert smoke < digest
    assert 'version("catanatron-rs") == "0.1.8"' in script
    assert "gumbel_search_capabilities" in script
    assert "sigma_reference_visits" in script
    assert "belief_target_evidence" in script
    assert "initial_road_d1_scope" in script
    assert "public_award_feature_parity" in script
    assert "policy_temperature_semantics" in script


def test_builder_runs_semantic_tests_for_advertised_corrected_capabilities() -> None:
    script = _script()
    build = script.index("maturin build")

    public_award = script.index(
        "entity_player_tokens_preserve_public_awards_when_hidden_hands_are_masked"
    )
    temperature = script.index("native/gumbel_mcts_rs/Cargo.toml")

    assert public_award < build
    assert temperature < build
    assert "--features python" in script[public_award - 180 : public_award]
    assert "temperature \\\n  --lib" in script[temperature : build]


def test_python_enabled_rust_test_resolves_relocated_uv_libpython() -> None:
    script = _script()
    public_award = script.index(
        "entity_player_tokens_preserve_public_awards_when_hidden_hands_are_masked"
    )
    prefix = script[:public_award]

    assert 'sysconfig.get_config_var("LDLIBRARY")' in prefix
    assert 'sysconfig.get_config_var("LIBDIR")' in prefix
    assert 'pathlib.Path(sys.base_prefix) / "lib"' in prefix
    assert 'LD_LIBRARY_PATH="$PYTHON_TEST_LIBDIR"' in prefix
    assert 'RUSTFLAGS="$RUSTFLAGS -L native=$PYTHON_TEST_LIBDIR"' in prefix
    # The shared-library workaround is scoped to the Python-enabled test.  It
    # must not become an ambient release compiler input.
    assert 'export LD_LIBRARY_PATH=' not in script
    assert prefix.rindex('LD_LIBRARY_PATH="$PYTHON_TEST_LIBDIR"') > prefix.rindex(
        "public_belief_determinization_tests"
    )

from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_zero.rl.production_recipe_catalog import (
    ProductionRecipeError,
    production_recipes,
    require_production_recipe,
)
from catan_zero.rl import production_recipe_catalog as catalog
from catan_zero.rl.pipeline_configs import config_from_payload


ROOT = Path(__file__).resolve().parents[1]
APPROVED = {
    "generate": (
        "configs/generation/coherent_public_n128.schema19.json",
        "coherent-public-n128",
    ),
    "evaluate": (
        "configs/eval/coherent_public_n128.schema19.json",
        "coherent-public-n128",
    ),
    "train": (
        "configs/training/a1_current_35m_b200.schema1.json",
        "a1-current-35m-b200",
    ),
}
APPROVED_PARENT_UPDATE = (
    "configs/training/a1_parent_update_35m_b200.schema1.json",
    "a1-parent-update-35m-b200",
)


@pytest.mark.parametrize("entrypoint", sorted(APPROVED))
def test_checked_in_production_recipe_is_authenticated(entrypoint: str) -> None:
    relative, expected_name = APPROVED[entrypoint]
    path = ROOT / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert (
        require_production_recipe(entrypoint=entrypoint, path=path, payload=payload)
        == expected_name
    )


def test_checked_in_parent_update_recipe_is_authenticated() -> None:
    relative, expected_name = APPROVED_PARENT_UPDATE
    path = ROOT / relative
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert (
        require_production_recipe(entrypoint="train", path=path, payload=payload)
        == expected_name
    )


def test_generation_recipe_round_trips_every_typed_science_field() -> None:
    path = ROOT / APPROVED["generate"][0]
    payload = json.loads(path.read_text(encoding="utf-8"))

    resolved = config_from_payload(payload)

    assert resolved.field_values() == payload["fields"]
    assert resolved.preserve_root_prior_value is True


def test_generation_guard_is_authenticated_by_the_same_catalog() -> None:
    entry = production_recipes("generate")[0]

    assert Path(entry["guard"]).is_absolute()
    assert entry["guard_sha256"] == (
        "ca2909a0f725b0af82f144ab1cc1b2db2b42b4d30676304031066842b2ded5a8"
    )


def test_generation_guard_drift_is_rejected_from_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied_root = tmp_path / "repo"
    for relative in (
        "configs/production_recipes.json",
        APPROVED["generate"][0],
        "configs/guards/a1_generation_coherent_public_n128_adaptive256_forced_value_v3.json",
    ):
        source = ROOT / relative
        destination = copied_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    guard = (
        copied_root
        / "configs/guards/a1_generation_coherent_public_n128_adaptive256_forced_value_v3.json"
    )
    payload = json.loads(guard.read_text(encoding="utf-8"))
    payload["schema_version"] = "drifted"
    guard.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(catalog, "_repository_root", lambda: copied_root)

    with pytest.raises(ProductionRecipeError, match="guard bytes drifted"):
        production_recipes("generate")


def test_authenticated_catalog_listing_has_no_second_identity_registry() -> None:
    train = production_recipes("train")

    assert [entry["name"] for entry in train] == [
        "a1-current-35m-b200",
        "a1-parent-update-35m-b200",
    ]
    assert all(Path(entry["path"]).is_absolute() for entry in train)
    assert all(len(entry["canonical_sha256"]) == 64 for entry in train)


def test_unlisted_copy_cannot_enter_production(tmp_path: Path) -> None:
    source = ROOT / APPROVED["generate"][0]
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    with pytest.raises(ProductionRecipeError, match="checked-in regular file"):
        require_production_recipe(
            entrypoint="generate",
            path=copied,
            payload=json.loads(copied.read_text(encoding="utf-8")),
        )


def test_cataloged_path_with_drifted_payload_is_rejected() -> None:
    path = ROOT / APPROVED["evaluate"][0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fields"]["n_full"] = 256
    with pytest.raises(ProductionRecipeError, match="recipe bytes drifted"):
        require_production_recipe(entrypoint="evaluate", path=path, payload=payload)

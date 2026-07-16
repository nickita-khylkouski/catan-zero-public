from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_zero.rl.production_recipe_catalog import (
    ProductionRecipeError,
    production_recipes,
    require_production_recipe,
)


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
        require_production_recipe(
            entrypoint=entrypoint, path=path, payload=payload
        )
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

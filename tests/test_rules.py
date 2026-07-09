from pathlib import Path

from catan_zero.rules import load_rules


def test_rules_file_loads() -> None:
    rules = load_rules(Path(__file__).parents[1] / "catan_rules_v1.json")

    assert rules["ruleset_id"] == "CatanBench-4P-Full-v1"
    assert rules["players"] == 4
    assert rules["trading"]["structured_offers_only"] is True


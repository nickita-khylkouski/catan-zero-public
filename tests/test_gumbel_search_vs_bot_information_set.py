from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from gumbel_search_vs_bot_h2h import _search_config_kwargs  # noqa: E402


def test_vs_bot_search_threads_information_set_recipe() -> None:
    config = _search_config_kwargs(
        {
            "n_full": 128,
            "max_depth": 80,
            "correct_rust_chance_spectra": True,
            "information_set_search": True,
            "determinization_particles": 4,
            "determinization_min_simulations": 32,
        }
    )
    assert config["information_set_search"] is True
    assert config["determinization_particles"] == 4
    assert config["determinization_min_simulations"] == 32

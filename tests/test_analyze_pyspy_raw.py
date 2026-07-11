from tools.profiling.analyze_pyspy_raw import _category


def test_native_search_is_not_mislabeled_python_traversal() -> None:
    native = ";".join(
        (
            "_search_information_set (src/catan_zero/search/gumbel_chance_mcts.py:893)",
            "_search_single_world (src/catan_zero/search/native_gumbel_mcts.py:197)",
            "gumbel_mcts::simulate (site-packages/catanatron_rs.so)",
        )
    )
    orchestration = ";".join(
        (
            "_search_information_set (src/catan_zero/search/gumbel_chance_mcts.py:893)",
            "search (src/catan_zero/search/native_gumbel_mcts.py:225)",
        )
    )
    reference = "_simulate (src/catan_zero/search/gumbel_chance_mcts.py:1200)"

    assert _category("thread: MainThread", native) == "native_mcts_traversal_and_allocator"
    assert _category("thread: MainThread", orchestration) == "python_pimc_orchestration"
    assert _category("thread: MainThread", reference) == "python_mcts_traversal"

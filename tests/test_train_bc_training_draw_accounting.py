from tools.train_bc import _training_draw_accounting


def test_training_draw_accounting_separates_base_and_auxiliary_draws() -> None:
    report = _training_draw_accounting(
        [
            {"samples": 4_096, "policy_aux_active_rows": 128},
            {"samples": 2_048, "policy_aux_active_rows": 64},
        ]
    )

    assert report["training_row_draws"] == 6_144
    assert report["base_training_row_draws"] == 6_144
    assert report["policy_aux_training_row_draws"] == 192
    assert report["total_training_row_draws"] == 6_336
    assert report["unique_training_rows_drawn"] is None
    assert "may repeat rows" in report["training_row_draws_semantics"]


def test_training_draw_accounting_defaults_missing_auxiliary_dose_to_zero() -> None:
    report = _training_draw_accounting([{"samples": 512}])

    assert report["training_row_draws"] == 512
    assert report["policy_aux_training_row_draws"] == 0
    assert report["total_training_row_draws"] == 512

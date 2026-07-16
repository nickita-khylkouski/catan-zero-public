from __future__ import annotations

import pytest

from tools import a1_b200_stage_c_learner_campaign as campaign


def test_uniform_unique_coverage_contract_is_nontrivial_for_stage_c_dose() -> None:
    contract = campaign._uniform_unique_coverage_contract(  # noqa: SLF001
        population=65_536,
        draws=campaign.POLICY_AUX_ACTIVE_BATCH_SIZE
        * campaign.WORLD_SIZE
        * campaign.MAX_STEPS,
    )
    assert contract["expected_unique_roots"] > 14_000
    assert contract["lower_bound_unique_roots"] > 13_000
    assert contract["lower_bound_unique_roots"] < contract["expected_unique_roots"]


def test_strategic_balanced_sampling_surface_requires_exact_uniformity() -> None:
    campaign._verify_strategic_balanced_sampling_surface(  # noqa: SLF001
        {
            "final_training_weights": {
                "min": 1.0,
                "max": 1.0,
                "mean": 1.0,
                "effective_sample_size": 65_536.0,
            }
        },
        selected_training_roots=65_536,
    )
    with pytest.raises(campaign.CampaignError, match="not uniform"):
        campaign._verify_strategic_balanced_sampling_surface(  # noqa: SLF001
            {
                "final_training_weights": {
                    "min": 0.5,
                    "max": 1.5,
                    "mean": 1.0,
                    "effective_sample_size": 60_000.0,
                }
            },
            selected_training_roots=65_536,
        )


def test_strategic_balanced_realized_coverage_refuses_sampler_collapse() -> None:
    with pytest.raises(campaign.CampaignError, match="sampler collapsed"):
        campaign._verify_realized_policy_aux_coverage(  # noqa: SLF001
            arm="STRATEGIC_BALANCED",
            selected_training_roots=65_536,
            auxiliary_draws=16_384,
            unique_source_rows=1,
        )
    evidence = campaign._verify_realized_policy_aux_coverage(  # noqa: SLF001
        arm="STRATEGIC_BALANCED",
        selected_training_roots=65_536,
        auxiliary_draws=16_384,
        unique_source_rows=14_500,
    )
    assert evidence["gate_passed"] is True
    assert evidence["realized_unique_roots"] == 14_500

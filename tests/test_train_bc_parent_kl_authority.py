from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import train_bc


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _binding(*, producer: str, parent: str, initializer: str) -> dict[str, object]:
    return {
        "producer_checkpoint_sha256": producer,
        "learner_parent_checkpoint_sha256": parent,
        "learner_initializer_sha256": initializer,
    }


def test_adaptive_kl_refuses_stored_priors_from_nonparent_producer() -> None:
    producer = _sha("a")
    parent = _sha("b")
    initializer = _sha("c")

    with pytest.raises(
        SystemExit,
        match="stored prior_policy rows come from corpus producer",
    ) as error:
        train_bc._validate_parent_policy_kl_authority(  # noqa: SLF001
            SimpleNamespace(policy_kl_target=0.027),
            _binding(
                producer=producer,
                parent=parent,
                initializer=initializer,
            ),
        )

    message = str(error.value)
    assert producer in message
    assert parent in message
    assert initializer in message
    assert "must not be reported as parent KL" in message


def test_adaptive_kl_accepts_function_preserving_initializer_of_exact_parent() -> None:
    parent = _sha("d")
    initializer = _sha("e")

    authority = train_bc._validate_parent_policy_kl_authority(  # noqa: SLF001
        SimpleNamespace(policy_kl_target=0.012),
        _binding(
            producer=parent,
            parent=parent,
            initializer=initializer,
        ),
    )

    assert authority == {
        "schema_version": "train-bc-parent-policy-kl-authority-v1",
        "status": "verified_exact_parent",
        "stored_prior_checkpoint_sha256": parent,
        "parent_checkpoint_sha256": parent,
        "initializer_checkpoint_sha256": initializer,
        "function_preserving_initializer_allowed": True,
    }


@pytest.mark.parametrize(
    ("binding", "expected_producer", "expected_parent"),
    [
        (None, None, None),
        ({"producer_checkpoint_sha256": _sha("1")}, _sha("1"), None),
        ({"learner_parent_checkpoint_sha256": _sha("2")}, None, _sha("2")),
    ],
)
def test_unproven_parent_identity_is_explicitly_stored_prior_diagnostic(
    binding: dict[str, object] | None,
    expected_producer: str | None,
    expected_parent: str | None,
) -> None:
    authority = train_bc._validate_parent_policy_kl_authority(  # noqa: SLF001
        SimpleNamespace(policy_kl_target=0.01, init_checkpoint_sha256=None), binding
    )
    assert authority is not None
    assert authority["status"] == "stored_prior_not_verified_as_parent"
    assert authority["metric_semantics"] == (
        "stored_prior_policy_kl_not_parent_policy_kl"
    )
    assert authority["diagnostic_only"] is True
    assert authority["stored_prior_checkpoint_sha256"] == expected_producer
    assert authority["parent_checkpoint_sha256"] == expected_parent


def test_disabled_adaptive_kl_needs_no_parent_authority() -> None:
    assert (
        train_bc._validate_parent_policy_kl_authority(  # noqa: SLF001
            SimpleNamespace(policy_kl_target=None),
            _binding(producer=_sha("1"), parent=_sha("2"), initializer=_sha("3")),
        )
        is None
    )


def test_generic_binding_uses_init_checkpoint_as_declared_parent() -> None:
    parent = _sha("f")
    authority = train_bc._validate_parent_policy_kl_authority(  # noqa: SLF001
        SimpleNamespace(policy_kl_target=0.01, init_checkpoint_sha256=parent),
        {"producer_checkpoint_sha256": parent},
    )
    assert authority is not None
    assert authority["status"] == "verified_exact_parent"
    assert authority["parent_checkpoint_sha256"] == parent


def test_adaptive_kl_refuses_malformed_authenticated_authority() -> None:
    with pytest.raises(SystemExit, match="malformed checkpoint identity"):
        train_bc._validate_parent_policy_kl_authority(  # noqa: SLF001
            SimpleNamespace(policy_kl_target=0.01),
            _binding(
                producer=_sha("a"),
                parent="not-a-sha256",
                initializer=_sha("c"),
            ),
        )


def test_value_trunk_routing_attests_all_attention_pool_shared_inputs() -> None:
    routing = train_bc._value_trunk_gradient_routing(  # noqa: SLF001
        SimpleNamespace(
            value_trunk_grad_scale=0.25,
            arch="entity_graph",
            value_head_type="mse",
        ),
        scalar_weight=1.0,
        model=SimpleNamespace(value_attention_pool=True),
    )

    assert routing["scope"] == "scalar_value_readout_all_shared_inputs"
    assert routing["shared_input_paths"] == [
        "cls_state",
        "attention_pool_state",
        "attention_pool_tokens",
    ]
    assert routing["all_scalar_value_shared_inputs_scaled"] is True

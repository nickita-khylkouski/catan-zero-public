# Coherent-target identity R&D corpus v2

This version preserves the v1 coherent n128 target-identity experiment while
adding one training-signal invariant: every complete game must retain its
single-legal-action transitions as value-only rows. These rows have zero
policy weight and full value weight.

Authenticate the contract before launch:

```bash
python tools/a1_target_eligibility_inventory.py \
  --rd-contract configs/operations/a1-target-identity-coherent-n128-rd-v2/contract.json \
  --out /tmp/a1-coherent-target-rd-v2.contract-inventory.json
```

The v2 seed interval is disjoint from issued v1. Admission fails unless the
inventory proves forced-row coverage in all 8,192 games with
`policy_weight_multiplier=0` and `value_weight_multiplier=1`.

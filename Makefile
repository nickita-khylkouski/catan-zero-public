# CAT-129 canonical test/CI gate.
# `make test` = the ONE mandatory pre-deploy + post-deploy (CAT-130) check.
# Exit 0 iff: full suite green (GPU tests self-skip; CAT-94 concat + modal
# quarantined) AND champion no-op BIT-IDENTICAL AND featurizer parity 19/19
# AND CAT-75 CLI goldens. See docs/TESTGATE.md.

SHELL := /bin/bash

.PHONY: test gate suite parity noop help

test gate:
	bash scripts/gate.sh

# Sub-targets for debugging individual gate stages (not the mandatory gate):
suite:
	bash scripts/gate.sh --only suite

parity:
	bash scripts/gate.sh --only parity

noop:
	bash scripts/gate.sh --only noop

help:
	@echo "make test   — run the full canonical gate (suite + no-op + parity + goldens), exit 0 iff all pass"
	@echo "make suite  — full pytest suite only"
	@echo "make parity — featurizer parity 19/19 only"
	@echo "make noop    — champion no-op bit-identical only"

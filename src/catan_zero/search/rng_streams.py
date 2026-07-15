"""Deterministic domain separation for search randomness.

The legacy search intentionally uses one ``random.Random`` stream for every
draw.  Reliability probes need independent replications without allowing a
belief-world draw, a chance-node draw, or root Gumbel noise to steal entropy
from either of the other streams.  Keep the derivation tiny and explicit so
the exact contract can be written into generation manifests.
"""

from __future__ import annotations

import hashlib


SEARCH_RNG_STREAM_SCHEMA = "gumbel-chance-belief-domain-separation-v1"
SEARCH_RNG_STREAM_NAMES = ("gumbel", "chance", "belief")


def domain_separated_search_seed(base_seed: int, stream: str) -> int:
    """Return a stable unsigned-64 seed for one named search substream."""

    if stream not in SEARCH_RNG_STREAM_NAMES:
        raise ValueError(
            f"unknown search RNG stream {stream!r}; "
            f"expected one of {SEARCH_RNG_STREAM_NAMES!r}"
        )
    payload = f"{SEARCH_RNG_STREAM_SCHEMA}:{int(base_seed)}:{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

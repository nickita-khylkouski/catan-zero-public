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
SHARED_SEARCH_RNG_STREAM_SCHEMA = "gumbel-chance-belief-shared-stream-v1"
SEARCH_RNG_STREAM_NAMES = ("gumbel", "chance", "belief")
BOUNDARY_VALUE_PARTICLE_SCHEMA = "coherent-boundary-value-particle-v1"


def domain_separated_search_seed(base_seed: int, stream: str) -> int:
    """Return a stable unsigned-64 seed for one named search substream."""

    if stream not in SEARCH_RNG_STREAM_NAMES:
        raise ValueError(
            f"unknown search RNG stream {stream!r}; "
            f"expected one of {SEARCH_RNG_STREAM_NAMES!r}"
        )
    payload = f"{SEARCH_RNG_STREAM_SCHEMA}:{int(base_seed)}:{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def boundary_value_particle_seed(
    base_seed: int,
    *,
    root_index: int,
    particle_index: int,
) -> int:
    """Return one stateless boundary-world seed for a coherent-search root.

    The root index advances once per search call, independently of particle
    count. K=2 is therefore an exact sample prefix of K=4 at every matched
    root, and neither arm consumes the Gumbel/chance/native-engine RNG stream.
    """

    if int(root_index) < 0:
        raise ValueError("root_index must be non-negative")
    if int(particle_index) < 0:
        raise ValueError("particle_index must be non-negative")
    payload = (
        f"{BOUNDARY_VALUE_PARTICLE_SCHEMA}:{int(base_seed)}:"
        f"{int(root_index)}:{int(particle_index)}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

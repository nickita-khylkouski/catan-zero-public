"""RGSC ranking-based regret-prioritization sampler (task #64 follow-up, CAT-43).

Implements the archive-sampling rule from *Regret-Guided Search Control*
(RGSC; Tsai et al., ICLR 2026, https://github.com/rlglab/rgsc), which continued
improving a nearly-converged 9x9 Go model 69.3% -> 78.2% where both vanilla
AlphaZero and Go-Exploit's uniform archive sampling flatlined.

Paper mechanism (Section 3.3, "Prioritized Regret Buffer for Search
Control"): the buffer samples a state with probability

    P(s_i) = R(s_i)^(1/tau) / sum_j R(s_j)^(1/tau)                        (paper eq. in 3.3)

where R(s) is the state's regret and tau is a sampling temperature (the
reference repo's `env_buf_sampling_temperature` config default is 0.1). This
is a softmax-like power-law over regret VALUES, not a literal `1/rank(s)`
rule -- higher-regret states get higher probability, but the temperature
keeps the distribution soft (not deterministic top-K), which is what buys
RGSC its diversity advantage over Go-Exploit's flat-uniform archive sampling.

We reuse this rule directly. The paper additionally trains a *regret ranking
network* (Section 3.2) because directly regressing a state's true regret
value online is a hard, non-stationary, imbalanced learning target; our
regret score (`regret_common.score_shard`'s `regret_score`) is instead a
deterministic, offline, feature-based extraction score over an already
-finished corpus, so it has none of the pathologies the ranking-network
relaxation exists to fix -- R(s) here is the additive regret_score directly,
no separate ranking network needed. "Ranking-based" in this module's context
means: states are prioritized by where their regret_score RANKS in the
archive (via the power-law weighting above), rather than sampled uniformly
regardless of rank (Go-Exploit's approach, and this codebase's prior
default).

Sampling is WITHOUT replacement, via Efraimidis-Spirakis weighted reservoir
sampling: draw key_i = u_i^(1/w_i) for u_i ~ Uniform(0, 1) and weight w_i,
then keep the N largest keys. This is an exact, unbiased weighted sample
without replacement (Efraimidis & Spirakis, 2006) and needs no cumulative-sum
bookkeeping, so it composes directly with a `np.random.Generator`.
"""

from __future__ import annotations

import numpy as np

# Matches the RGSC reference repo's `env_buf_sampling_temperature` default
# (tools/quick-run.sh config docs: "temperature for buffer sampling; the
# default is 0.1").
DEFAULT_RGSC_TEMPERATURE = 0.1

# Regret scores are a non-negative additive sum of ~[0,1] components
# (regret_common.score_shard); scores of exactly 0 would otherwise make
# R(s)^(1/tau) collapse to 0 for every state that has zero regret, so a
# small floor keeps every candidate reachable (never truly excluded) while
# leaving the ranking of every positive-regret state untouched.
_MIN_WEIGHT_EPS = 1e-9


def rgsc_weights(
    regret_scores: np.ndarray, *, temperature: float = DEFAULT_RGSC_TEMPERATURE
) -> np.ndarray:
    """Unnormalised RGSC sampling weights `R(s)^(1/tau)` for an array of regret scores.

    `temperature` must be > 0 (lower tau => more deterministic / top-heavy;
    tau -> 1 approaches raw-score-proportional; tau -> 0+ approaches greedy
    top-1). Scores are floored at `_MIN_WEIGHT_EPS` before the power so a
    zero-regret state gets a tiny nonzero weight rather than being an
    absolute impossibility.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    scores = np.clip(np.asarray(regret_scores, dtype=np.float64), _MIN_WEIGHT_EPS, None)
    return np.power(scores, 1.0 / temperature)


def rgsc_probabilities(
    regret_scores: np.ndarray, *, temperature: float = DEFAULT_RGSC_TEMPERATURE
) -> np.ndarray:
    """Normalised RGSC sampling distribution `P(s_i)` over `regret_scores`."""
    weights = rgsc_weights(regret_scores, temperature=temperature)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        # Degenerate corpus (e.g. all-zero regret): fall back to uniform.
        return np.full(weights.shape, 1.0 / weights.size, dtype=np.float64)
    return weights / total


def rgsc_sample_indices(
    regret_scores: np.ndarray,
    k: int,
    *,
    temperature: float = DEFAULT_RGSC_TEMPERATURE,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample `min(k, n)` indices into `regret_scores` without replacement,
    weighted by the RGSC power-law rule (see module docstring).

    Uses Efraimidis-Spirakis weighted reservoir sampling: assigns each
    candidate a key `u_i^(1/w_i)` (u_i ~ Uniform(0,1)) and keeps the
    `k` largest keys. Higher-regret states get systematically higher keys in
    expectation, so the result is priority-biased, but any positive-regret
    state can still be drawn (unlike a deterministic top-K slice).
    """
    n = int(np.asarray(regret_scores).shape[0])
    k = max(0, min(int(k), n))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    weights = rgsc_weights(regret_scores, temperature=temperature)
    u = rng.random(n)
    # log-space to avoid overflow for large weights: log(key) = log(u) / w.
    # u in (0, 1) is guaranteed by np.random.Generator.random's [0, 1) support
    # union a floor, since log(0) is -inf.
    u = np.clip(u, np.finfo(np.float64).tiny, None)
    log_keys = np.log(u) / weights
    # Largest log-key == largest key (monotonic transform); argpartition for
    # the top-k, then order by key descending for a deterministic tie-break.
    if k >= n:
        order = np.argsort(-log_keys, kind="stable")
        return order.astype(np.int64)
    top = np.argpartition(-log_keys, k - 1)[:k]
    return top[np.argsort(-log_keys[top], kind="stable")].astype(np.int64)


def uniform_sample_indices(n: int, k: int, *, rng: np.random.Generator) -> np.ndarray:
    """Sample `min(k, n)` indices in `[0, n)` uniformly without replacement.

    Thin wrapper kept alongside `rgsc_sample_indices` so callers can switch
    sampling strategy via one function reference (`--restart-sampling`).
    """
    n = int(n)
    k = max(0, min(int(k), n))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(n, size=k, replace=False).astype(np.int64)


def mean_regret_by_rank_bucket(
    regret_scores: np.ndarray, selected: np.ndarray, *, n_buckets: int = 4
) -> list[float]:
    """Diagnostic: mean regret_score of `selected` indices, split by which
    rank-quartile (1 = highest regret) of `regret_scores` they fall in.

    Used for the CAT-43 smoke-test verification step ("manually check that
    top-ranked archived states do have higher measured regret than
    bottom-ranked ones") -- a sane `rgsc`-mode sample should show a
    decreasing mean regret_score from bucket 0 (top ranks) to bucket
    `n_buckets - 1` (bottom ranks).
    """
    scores = np.asarray(regret_scores, dtype=np.float64)
    n = scores.shape[0]
    # rank 0 = highest score (descending regret order).
    order = np.argsort(-scores, kind="stable")
    rank_of = np.empty(n, dtype=np.int64)
    rank_of[order] = np.arange(n)
    bucket_of = np.clip((rank_of * n_buckets) // max(n, 1), 0, n_buckets - 1)
    out = []
    for b in range(n_buckets):
        mask = bucket_of[np.asarray(selected, dtype=np.int64)] == b
        vals = scores[np.asarray(selected, dtype=np.int64)][mask]
        out.append(float(vals.mean()) if vals.size else float("nan"))
    return out

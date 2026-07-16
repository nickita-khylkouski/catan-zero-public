"""Strict, content-addressed configuration for canonical distributed PPO runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA = "canonical_entity_ppo_run_v2"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ManifestError(ValueError):
    """The PPO manifest is incomplete, malformed, or internally inconsistent."""


def _object(value: Any, *, where: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{where} must be an object")
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise ManifestError(f"{where} keys differ: missing={missing} unknown={unknown}")
    return value


def _string(value: Any, *, where: str, choices: set[str] | None = None) -> str:
    if type(value) is not str:  # exact: bool/numeric coercion is forbidden
        raise ManifestError(f"{where} must be a string")
    if not value:
        raise ManifestError(f"{where} must not be empty")
    if choices is not None and value not in choices:
        raise ManifestError(f"{where} must be one of {sorted(choices)}, got {value!r}")
    return value


def _integer(value: Any, *, where: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ManifestError(f"{where} must be an integer")
    if value < minimum:
        raise ManifestError(f"{where} must be >= {minimum}")
    return value


def _number(
    value: Any,
    *,
    where: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float:
        raise ManifestError(f"{where} must be a JSON floating-point number")
    if not math.isfinite(value):
        raise ManifestError(f"{where} must be finite")
    if minimum is not None and value < minimum:
        raise ManifestError(f"{where} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ManifestError(f"{where} must be <= {maximum}")
    return value


def _boolean(value: Any, *, where: str) -> bool:
    if type(value) is not bool:
        raise ManifestError(f"{where} must be a boolean")
    return value


def _strings(value: Any, *, where: str, allow_empty: bool = False) -> tuple[str, ...]:
    if type(value) is not list:
        raise ManifestError(f"{where} must be a list")
    result = tuple(
        _string(item, where=f"{where}[{index}]") for index, item in enumerate(value)
    )
    if not allow_empty and not result:
        raise ManifestError(f"{where} must not be empty")
    if len(set(result)) != len(result):
        raise ManifestError(f"{where} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class PPOIdentity:
    track: str
    vps_to_win: int
    architecture: str
    initializer_sha256: str

    @classmethod
    def from_dict(cls, raw: Any) -> "PPOIdentity":
        value = _object(raw, where="spec.identity", keys=set(cls.__annotations__))
        digest = _string(
            value["initializer_sha256"], where="spec.identity.initializer_sha256"
        )
        if not _SHA256.fullmatch(digest):
            raise ManifestError(
                "spec.identity.initializer_sha256 must be sha256:<64 lowercase hex>"
            )
        return cls(
            track=_string(
                value["track"], where="spec.identity.track", choices={"2p_no_trade"}
            ),
            vps_to_win=_integer(
                value["vps_to_win"], where="spec.identity.vps_to_win", minimum=1
            ),
            architecture=_string(
                value["architecture"],
                where="spec.identity.architecture",
                choices={"entity_graph"},
            ),
            initializer_sha256=digest,
        )


@dataclass(frozen=True, slots=True)
class PPOActorSpec:
    max_decisions: int
    games_per_shard: int
    gamma: float
    gae_lambda: float
    action_temperature: float
    value_shaping_coef: float
    value_shaping_scale: float
    value_shaping_opponent_penalty: float
    seed: int
    seat_schedule: str
    opponent_mode: str
    opponents: tuple[str, ...]
    pfsp_mode: str

    @classmethod
    def from_dict(cls, raw: Any) -> "PPOActorSpec":
        value = _object(raw, where="spec.actor", keys=set(cls.__annotations__))
        return cls(
            max_decisions=_integer(
                value["max_decisions"], where="spec.actor.max_decisions", minimum=1
            ),
            games_per_shard=_integer(
                value["games_per_shard"], where="spec.actor.games_per_shard", minimum=1
            ),
            gamma=_number(
                value["gamma"], where="spec.actor.gamma", minimum=0.0, maximum=1.0
            ),
            gae_lambda=_number(
                value["gae_lambda"],
                where="spec.actor.gae_lambda",
                minimum=0.0,
                maximum=1.0,
            ),
            action_temperature=_number(
                value["action_temperature"],
                where="spec.actor.action_temperature",
                minimum=1e-12,
            ),
            value_shaping_coef=_number(
                value["value_shaping_coef"],
                where="spec.actor.value_shaping_coef",
                minimum=0.0,
            ),
            value_shaping_scale=_number(
                value["value_shaping_scale"],
                where="spec.actor.value_shaping_scale",
                minimum=1e-12,
            ),
            value_shaping_opponent_penalty=_number(
                value["value_shaping_opponent_penalty"],
                where="spec.actor.value_shaping_opponent_penalty",
                minimum=0.0,
            ),
            seed=_integer(value["seed"], where="spec.actor.seed"),
            seat_schedule=_string(
                value["seat_schedule"],
                where="spec.actor.seat_schedule",
                choices={"round_robin"},
            ),
            opponent_mode=_string(
                value["opponent_mode"],
                where="spec.actor.opponent_mode",
                choices={"fixed", "league"},
            ),
            opponents=_strings(value["opponents"], where="spec.actor.opponents"),
            pfsp_mode=_string(
                value["pfsp_mode"], where="spec.actor.pfsp_mode", choices={"pfsp"}
            ),
        )


@dataclass(frozen=True, slots=True)
class PPOLearnerSpec:
    shards_per_step: int
    max_staleness: int
    max_steps: int
    resume: bool
    lr: float
    trunk_lr_mult: float
    clip_ratio: float
    value_coef: float
    value_clip_range: float
    entropy_coef: float
    ppo_epochs: int
    minibatch_size: int
    target_kl: float
    top_advantage_fraction: float
    min_advantage_samples: int
    advantage_normalization: str
    advantage_group_weights: tuple[str, ...]
    kl_to_bc_init: float
    kl_to_bc_final: float
    kl_to_bc_anneal_steps: int
    use_vtrace: bool
    vtrace_clip_rho: float
    vtrace_clip_pg_rho: float
    vtrace_use_current_values: bool
    vtrace_forward_chunk: int

    @classmethod
    def from_dict(cls, raw: Any) -> "PPOLearnerSpec":
        value = _object(raw, where="spec.learner", keys=set(cls.__annotations__))
        result = cls(
            shards_per_step=_integer(
                value["shards_per_step"],
                where="spec.learner.shards_per_step",
                minimum=1,
            ),
            max_staleness=_integer(
                value["max_staleness"], where="spec.learner.max_staleness"
            ),
            max_steps=_integer(value["max_steps"], where="spec.learner.max_steps"),
            resume=_boolean(value["resume"], where="spec.learner.resume"),
            lr=_number(value["lr"], where="spec.learner.lr", minimum=1e-16),
            trunk_lr_mult=_number(
                value["trunk_lr_mult"],
                where="spec.learner.trunk_lr_mult",
                minimum=1e-16,
                maximum=1.0,
            ),
            clip_ratio=_number(
                value["clip_ratio"],
                where="spec.learner.clip_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            value_coef=_number(
                value["value_coef"], where="spec.learner.value_coef", minimum=0.0
            ),
            value_clip_range=_number(
                value["value_clip_range"],
                where="spec.learner.value_clip_range",
                minimum=0.0,
            ),
            entropy_coef=_number(
                value["entropy_coef"], where="spec.learner.entropy_coef", minimum=0.0
            ),
            ppo_epochs=_integer(
                value["ppo_epochs"], where="spec.learner.ppo_epochs", minimum=1
            ),
            minibatch_size=_integer(
                value["minibatch_size"], where="spec.learner.minibatch_size", minimum=1
            ),
            target_kl=_number(
                value["target_kl"], where="spec.learner.target_kl", minimum=0.0
            ),
            top_advantage_fraction=_number(
                value["top_advantage_fraction"],
                where="spec.learner.top_advantage_fraction",
                minimum=1e-12,
                maximum=1.0,
            ),
            min_advantage_samples=_integer(
                value["min_advantage_samples"],
                where="spec.learner.min_advantage_samples",
                minimum=1,
            ),
            advantage_normalization=_string(
                value["advantage_normalization"],
                where="spec.learner.advantage_normalization",
                choices={"global", "per_opponent", "none"},
            ),
            advantage_group_weights=_strings(
                value["advantage_group_weights"],
                where="spec.learner.advantage_group_weights",
                allow_empty=True,
            ),
            kl_to_bc_init=_number(
                value["kl_to_bc_init"], where="spec.learner.kl_to_bc_init", minimum=0.0
            ),
            kl_to_bc_final=_number(
                value["kl_to_bc_final"],
                where="spec.learner.kl_to_bc_final",
                minimum=0.0,
            ),
            kl_to_bc_anneal_steps=_integer(
                value["kl_to_bc_anneal_steps"],
                where="spec.learner.kl_to_bc_anneal_steps",
            ),
            use_vtrace=_boolean(value["use_vtrace"], where="spec.learner.use_vtrace"),
            vtrace_clip_rho=_number(
                value["vtrace_clip_rho"],
                where="spec.learner.vtrace_clip_rho",
                minimum=1e-12,
                maximum=1.0,
            ),
            vtrace_clip_pg_rho=_number(
                value["vtrace_clip_pg_rho"],
                where="spec.learner.vtrace_clip_pg_rho",
                minimum=1e-12,
                maximum=1.0,
            ),
            vtrace_use_current_values=_boolean(
                value["vtrace_use_current_values"],
                where="spec.learner.vtrace_use_current_values",
            ),
            vtrace_forward_chunk=_integer(
                value["vtrace_forward_chunk"],
                where="spec.learner.vtrace_forward_chunk",
                minimum=1,
            ),
        )
        if not result.use_vtrace and result.max_staleness != 0:
            raise ManifestError(
                "spec.learner.max_staleness must be 0 when V-trace is disabled"
            )
        return result


@dataclass(frozen=True, slots=True)
class PPOCheckpointSpec:
    every_steps: int
    keep_last: int
    milestone_every: int

    @classmethod
    def from_dict(cls, raw: Any) -> "PPOCheckpointSpec":
        value = _object(raw, where="spec.checkpoint", keys=set(cls.__annotations__))
        return cls(
            every_steps=_integer(
                value["every_steps"], where="spec.checkpoint.every_steps", minimum=1
            ),
            keep_last=_integer(
                value["keep_last"], where="spec.checkpoint.keep_last", minimum=1
            ),
            milestone_every=_integer(
                value["milestone_every"],
                where="spec.checkpoint.milestone_every",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class PPOEvaluationSpec:
    dev_games: int
    promotion_games: int
    tracks: tuple[str, ...]
    opponents: tuple[str, ...]
    workers: int
    max_decisions: int
    device: str
    timeout_secs: float

    @classmethod
    def from_dict(cls, raw: Any) -> "PPOEvaluationSpec":
        value = _object(raw, where="spec.evaluation", keys=set(cls.__annotations__))
        tracks = _strings(value["tracks"], where="spec.evaluation.tracks")
        if any(track != "2p_no_trade" for track in tracks):
            raise ManifestError("spec.evaluation.tracks supports only '2p_no_trade'")
        return cls(
            dev_games=_integer(
                value["dev_games"], where="spec.evaluation.dev_games", minimum=1
            ),
            promotion_games=_integer(
                value["promotion_games"],
                where="spec.evaluation.promotion_games",
                minimum=1,
            ),
            tracks=tracks,
            opponents=_strings(value["opponents"], where="spec.evaluation.opponents"),
            workers=_integer(
                value["workers"], where="spec.evaluation.workers", minimum=1
            ),
            max_decisions=_integer(
                value["max_decisions"],
                where="spec.evaluation.max_decisions",
                minimum=1,
            ),
            device=_string(value["device"], where="spec.evaluation.device"),
            timeout_secs=_number(
                value["timeout_secs"],
                where="spec.evaluation.timeout_secs",
                minimum=1e-12,
            ),
        )


@dataclass(frozen=True, slots=True)
class PPOLeagueSpec:
    snapshot_interval: int
    promote_winrate: float

    @classmethod
    def from_dict(cls, raw: Any) -> "PPOLeagueSpec":
        value = _object(raw, where="spec.league", keys=set(cls.__annotations__))
        return cls(
            snapshot_interval=_integer(
                value["snapshot_interval"],
                where="spec.league.snapshot_interval",
                minimum=1,
            ),
            promote_winrate=_number(
                value["promote_winrate"],
                where="spec.league.promote_winrate",
                minimum=0.0,
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class PPORunSpec:
    identity: PPOIdentity
    actor: PPOActorSpec
    learner: PPOLearnerSpec
    checkpoint: PPOCheckpointSpec
    evaluation: PPOEvaluationSpec
    league: PPOLeagueSpec

    @classmethod
    def from_dict(cls, raw: Any) -> "PPORunSpec":
        value = _object(raw, where="spec", keys=set(cls.__annotations__))
        return cls(
            identity=PPOIdentity.from_dict(value["identity"]),
            actor=PPOActorSpec.from_dict(value["actor"]),
            learner=PPOLearnerSpec.from_dict(value["learner"]),
            checkpoint=PPOCheckpointSpec.from_dict(value["checkpoint"]),
            evaluation=PPOEvaluationSpec.from_dict(value["evaluation"]),
            league=PPOLeagueSpec.from_dict(value["league"]),
        )


@dataclass(frozen=True, slots=True)
class PPORunManifest:
    status: str
    spec: PPORunSpec

    @classmethod
    def from_dict(cls, raw: Any) -> "PPORunManifest":
        value = _object(raw, where="manifest", keys={"schema", "status", "spec"})
        schema = _string(value["schema"], where="schema")
        if schema != SCHEMA:
            raise ManifestError(f"schema must be {SCHEMA!r}, got {schema!r}")
        status = _string(value["status"], where="status", choices={"bound", "template"})
        spec = PPORunSpec.from_dict(value["spec"])
        if (
            status == "bound"
            and spec.identity.initializer_sha256 == "sha256:" + "0" * 64
        ):
            raise ManifestError("a bound manifest must name real initializer bytes")
        return cls(status=status, spec=spec)

    @classmethod
    def from_json(cls, raw: str) -> "PPORunManifest":
        try:
            value = json.loads(
                raw,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ManifestError(f"non-finite JSON constant is forbidden: {token}")
                ),
            )
        except json.JSONDecodeError as error:
            raise ManifestError(f"manifest is not valid JSON: {error}") from error
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "status": self.status, "spec": asdict(self.spec)}

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def sha256(self) -> str:
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


def load_manifest(path: str | Path) -> PPORunManifest:
    return PPORunManifest.from_json(Path(path).read_text(encoding="utf-8"))

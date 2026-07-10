#!/usr/bin/env python3
"""Run one R&D operator arm against a frozen reference and emit a result bundle.

The first supported arena is deliberately narrow: two-player, no player trade,
10-VP base Catan.  Every seed is played twice with candidate/reference colors
swapped.  Public-information search is required by default; authoritative
hidden-state diagnostics require an explicit command-line opt-in.

This module owns the complete boundary that ``operator_runner.py`` intentionally
does not: game construction, live authoritative chance resolution, paired seat
swaps, terminal-game enforcement, per-role counter accumulation, and immutable
provenance.  It writes nothing until every requested pair has completed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catan_zero.rl.entity_token_policy import EntityGraphPolicy  # noqa: E402
from catan_zero.rl.gumbel_self_play import _apply_selected_action  # noqa: E402
from catan_zero.search.gumbel_chance_mcts import (  # noqa: E402
    GumbelChanceMCTSConfig,
)
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.search.operator_runner import (  # noqa: E402
    GameCounterAccumulator,
    MeasuredDecision,
    MeasuredSearchOperator,
)
from catan_zero.search.regularized_mcts import RegularizedMCTSConfig  # noqa: E402
from catan_zero.search.rust_mcts import (  # noqa: E402
    RustMCTSConfig,
    _require_rust_module,
)


SCHEMA_VERSION = "catan-zero-rnd-leaderboard/v1"
SEED_MANIFEST_VERSION = "catan-zero-rnd-paired-seeds/v1"
TRAINING_MANIFEST_VERSION = "catan-zero-rnd-training-manifest/v1"
TRACK = "2p_no_trade"
COLORS: tuple[str, str] = ("RED", "BLUE")
AUTHORITATIVE_REGIME = "authoritative_hidden_state"
PUBLIC_REGIMES = frozenset(
    {"public_conservation_pimc", "public_observation_policy"}
)
SEARCH_KINDS = ("gumbel", "puct", "regularized_mcts", "raw_policy")


class BundleRunError(RuntimeError):
    """The paired campaign cannot emit a trustworthy result bundle."""


class MeasuredOperator(Protocol):
    information_regime: str

    def run(
        self, game: Any, *, require_public_information: bool = False
    ) -> MeasuredDecision: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ArmIdentity:
    arm_id: str
    architecture_id: str
    parameter_count: int
    architecture_config_sha256: str
    search_id: str
    search_config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str

    def candidate_payload(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "architecture": {
                "architecture_id": self.architecture_id,
                "parameter_count": int(self.parameter_count),
                "config_sha256": self.architecture_config_sha256,
            },
            "search": {
                "search_id": self.search_id,
                "config_sha256": self.search_config_sha256,
            },
            "checkpoint": {
                "path": self.checkpoint_path,
                "sha256": self.checkpoint_sha256,
            },
        }


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenReference:
    reference_id: str
    architecture_id: str
    parameter_count: int
    architecture_config_sha256: str
    search_id: str
    search_config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "architecture": {
                "architecture_id": self.architecture_id,
                "parameter_count": int(self.parameter_count),
                "config_sha256": self.architecture_config_sha256,
            },
            "search": {
                "search_id": self.search_id,
                "config_sha256": self.search_config_sha256,
            },
            "checkpoint": {
                "path": self.checkpoint_path,
                "sha256": self.checkpoint_sha256,
            },
        }


GameFactory = Callable[[int], Any]
ActionApplier = Callable[[Any, int, random.Random], Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def capture_git_provenance(repo: str | Path) -> dict[str, Any]:
    root = Path(repo).resolve()

    def git(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", *args], cwd=root, stderr=subprocess.PIPE
        )

    commit = git("rev-parse", "HEAD").decode("ascii").strip().lower()
    status = git("status", "--porcelain=v1", "-z")
    dirty = bool(status)
    payload: dict[str, Any] = {"git_commit": commit, "dirty": dirty}
    if not dirty:
        return payload

    digest = hashlib.sha256()
    digest.update(b"git-status-v1\0")
    digest.update(status)
    digest.update(b"git-diff-head-binary\0")
    digest.update(git("diff", "--binary", "HEAD", "--"))
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    for raw_relative in sorted(part for part in untracked.split(b"\0") if part):
        relative = raw_relative.decode("utf-8", errors="surrogateescape")
        path = root / relative
        digest.update(b"untracked\0")
        digest.update(raw_relative)
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"non-regular")
    payload["patch_sha256"] = digest.hexdigest()
    return payload


def load_seed_manifest(path: str | Path) -> tuple[list[int], dict[str, Any]]:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleRunError(f"cannot read paired-seed manifest {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BundleRunError("paired-seed manifest must be a JSON object")
    if payload.get("schema_version") != SEED_MANIFEST_VERSION:
        raise BundleRunError(
            f"paired-seed manifest schema_version must be {SEED_MANIFEST_VERSION!r}"
        )
    if payload.get("track") != TRACK:
        raise BundleRunError(f"paired-seed manifest track must be {TRACK!r}")
    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise BundleRunError("paired-seed manifest seeds must be a non-empty array")
    seeds: list[int] = []
    for index, seed in enumerate(raw_seeds):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise BundleRunError(f"paired-seed manifest seeds[{index}] must be an integer >= 0")
        seeds.append(int(seed))
    if len(set(seeds)) != len(seeds):
        raise BundleRunError("paired-seed manifest seeds must be unique")
    return seeds, {
        "path": str(source),
        "sha256": sha256_file(source),
        "schema_version": SEED_MANIFEST_VERSION,
        "track": TRACK,
        "seed_count": len(seeds),
    }


def load_training_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = _read_json_object(source, label="training manifest")
    if payload.get("schema_version") != TRAINING_MANIFEST_VERSION:
        raise BundleRunError(
            f"training manifest schema_version must be {TRAINING_MANIFEST_VERSION!r}"
        )
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "schema_version": TRAINING_MANIFEST_VERSION,
    }


def _default_game_factory(seed: int) -> Any:
    rust = _require_rust_module()
    return rust.Game.simple(list(COLORS), seed=int(seed))


def _default_action_applier(game: Any, action: int, rng: random.Random) -> Any:
    return _apply_selected_action(
        game,
        int(action),
        colors=COLORS,
        rng=rng,
        correct_rust_chance_spectra=True,
    )


def _reset_operator_for_game(operator: MeasuredOperator, seed: int) -> None:
    """Reset algorithm RNG without resetting evaluator/accounting state.

    Both seat orientations receive the same role-specific seed.  This prevents
    the first orientation from consuming the second orientation's search RNG
    stream.  Sync evaluators used by the CLI have cache_size=0, so no cross-game
    cache state changes measured work.
    """

    search = getattr(operator, "search", None)
    rng = getattr(search, "rng", None)
    if rng is not None and hasattr(rng, "seed"):
        rng.seed(int(seed))
    prepare = getattr(operator, "prepare_game", None)
    if callable(prepare):
        prepare(seed=int(seed))


def _regime(operator: MeasuredOperator) -> str:
    regime = str(getattr(operator, "information_regime", ""))
    if regime not in PUBLIC_REGIMES and regime != AUTHORITATIVE_REGIME:
        raise BundleRunError(f"operator reports unknown information regime {regime!r}")
    return regime


def play_measured_game(
    candidate: MeasuredOperator,
    reference: MeasuredOperator,
    *,
    game_seed: int,
    pair_id: str,
    candidate_seat: int,
    max_decisions: int,
    require_public_information: bool = True,
    game_factory: GameFactory = _default_game_factory,
    action_applier: ActionApplier = _default_action_applier,
) -> dict[str, Any]:
    if candidate_seat not in (0, 1):
        raise BundleRunError("candidate_seat must be 0 or 1")
    if int(max_decisions) < 1:
        raise BundleRunError("max_decisions must be positive")
    reference_seat = 1 - int(candidate_seat)
    candidate_color = COLORS[candidate_seat]
    reference_color = COLORS[reference_seat]
    operator_by_color = {candidate_color: candidate, reference_color: reference}
    candidate_counts = GameCounterAccumulator()
    reference_counts = GameCounterAccumulator()
    candidate_decisions = 0
    reference_decisions = 0

    # Candidate/reference seeds are role-specific but identical across the two
    # seat orientations for a given board seed.
    _reset_operator_for_game(candidate, int(game_seed) ^ 0x43414E44)
    _reset_operator_for_game(reference, int(game_seed) ^ 0x52454645)
    game = game_factory(int(game_seed))
    chance_rng = random.Random(int(game_seed) ^ 0xA17E)

    for _decision_index in range(int(max_decisions)):
        winner = game.winning_color()
        if winner is not None:
            break
        legal = tuple(
            int(action)
            for action in game.playable_action_indices(list(COLORS), None)
        )
        if not legal:
            raise BundleRunError(
                f"game seed={game_seed} seat={candidate_seat} has no legal actions before terminal"
            )
        acting_color = str(game.current_color())
        operator = operator_by_color.get(acting_color)
        if operator is None:
            raise BundleRunError(
                f"game seed={game_seed} produced unsupported acting color {acting_color!r}"
            )
        decision = operator.run(
            game,
            require_public_information=bool(require_public_information),
        )
        if int(decision.selected_action) not in legal:
            raise BundleRunError(
                f"operator selected illegal action {decision.selected_action}; legal={legal}"
            )
        if str(decision.information_regime) != _regime(operator):
            raise BundleRunError("decision information regime disagrees with its operator")
        if operator is candidate:
            candidate_counts.add(decision)
            candidate_decisions += 1
        else:
            reference_counts.add(decision)
            reference_decisions += 1
        game = action_applier(game, int(decision.selected_action), chance_rng)

    winner = game.winning_color()
    if winner is None:
        raise BundleRunError(
            f"game seed={game_seed} seat={candidate_seat} did not terminate "
            f"within max_decisions={max_decisions}"
        )
    winner_name = str(winner)
    if winner_name not in COLORS:
        raise BundleRunError(f"terminal game returned unsupported winner {winner_name!r}")

    candidate_regime = _regime(candidate)
    reference_regime = _regime(reference)
    if require_public_information and (
        candidate_regime not in PUBLIC_REGIMES or reference_regime not in PUBLIC_REGIMES
    ):
        raise BundleRunError("public-information campaign produced an authoritative-state game")
    return {
        "game_id": f"{pair_id}-candidate-seat-{candidate_seat}",
        "pair_id": str(pair_id),
        "seed": int(game_seed),
        "track": TRACK,
        "seat_assignment": {
            "candidate": int(candidate_seat),
            "reference": int(reference_seat),
        },
        "completed": True,
        "winner": winner_name,
        "candidate_score": 1.0 if winner_name == candidate_color else 0.0,
        "information_regime": candidate_regime,
        "reference_information_regime": reference_regime,
        "decisions": {
            "candidate": candidate_decisions,
            "reference": reference_decisions,
            "total": candidate_decisions + reference_decisions,
        },
        # The leaderboard compute contract compares the experimental candidate.
        "counters": candidate_counts.as_dict(),
        # Kept for auditability but deliberately not used to rank the candidate.
        "reference_counters": reference_counts.as_dict(),
    }


def build_result_bundle(
    candidate: MeasuredOperator,
    reference: MeasuredOperator,
    *,
    campaign_id: str,
    run_id: str,
    budget_regime: str,
    candidate_identity: ArmIdentity,
    reference_identity: FrozenReference,
    seeds: Sequence[int],
    seed_manifest: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    code_provenance: Mapping[str, Any],
    max_decisions: int,
    require_public_information: bool = True,
    game_factory: GameFactory = _default_game_factory,
    action_applier: ActionApplier = _default_action_applier,
) -> dict[str, Any]:
    if budget_regime not in ("equal_work", "equal_time"):
        raise BundleRunError("budget_regime must be equal_work or equal_time")
    if not str(campaign_id).strip() or not str(run_id).strip():
        raise BundleRunError("campaign_id and run_id must be non-empty")
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise BundleRunError("seeds must be non-empty and unique")
    required_regime = "public_only" if require_public_information else "allow_authoritative"
    if require_public_information:
        if _regime(candidate) not in PUBLIC_REGIMES or _regime(reference) not in PUBLIC_REGIMES:
            raise BundleRunError(
                "both candidate and reference must expose public-information adapters"
            )

    games: list[dict[str, Any]] = []
    for pair_index, game_seed in enumerate(normalized_seeds):
        pair_id = f"pair-{pair_index:06d}"
        pair_games = [
            play_measured_game(
                candidate,
                reference,
                game_seed=game_seed,
                pair_id=pair_id,
                candidate_seat=candidate_seat,
                max_decisions=max_decisions,
                require_public_information=require_public_information,
                game_factory=game_factory,
                action_applier=action_applier,
            )
            for candidate_seat in (0, 1)
        ]
        if pair_games[0]["seed"] != pair_games[1]["seed"]:
            raise BundleRunError("internal paired-seed mismatch")
        games.extend(pair_games)

    identity = candidate_identity.candidate_payload()
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": str(campaign_id),
        "run_id": str(run_id),
        "arm_id": identity["arm_id"],
        "budget_regime": budget_regime,
        "track": TRACK,
        "required_information_regime": required_regime,
        "architecture": identity["architecture"],
        "search": identity["search"],
        "checkpoint": identity["checkpoint"],
        "reference": reference_identity.payload(),
        "code": dict(code_provenance),
        "seed_manifest": dict(seed_manifest),
        "training_manifest": dict(training_manifest),
        "seed_protocol": {
            "board_and_chance": "same game_seed in both seat orientations",
            "candidate_search": "game_seed xor 0x43414E44, reset before each orientation",
            "reference_search": "game_seed xor 0x52454645, reset before each orientation",
        },
        "games": games,
    }


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleRunError(f"cannot read {label} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleRunError(f"{label} must be a JSON object")
    return value


def _load_campaign_arm(path: str | Path, arm_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = _read_json_object(path, label="campaign")
    if campaign.get("schema_version") != SCHEMA_VERSION:
        raise BundleRunError(f"campaign schema_version must be {SCHEMA_VERSION!r}")
    campaign_id = campaign.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise BundleRunError("campaign campaign_id must be a non-empty string")
    arms = campaign.get("arms")
    if not isinstance(arms, list):
        raise BundleRunError("campaign arms must be an array")
    matches = [arm for arm in arms if isinstance(arm, dict) and arm.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise BundleRunError(f"campaign must contain exactly one arm_id={arm_id!r}")
    arm = dict(matches[0])
    if arm.get("source_status") != "implemented":
        raise BundleRunError(f"campaign arm {arm_id!r} source is not implemented")
    requirement = campaign.get("required_information_regime")
    if requirement not in ("public_only", "allow_authoritative"):
        raise BundleRunError("campaign has invalid required_information_regime")
    return campaign, arm


def _assert_search_arm_binding(
    *,
    arm: Mapping[str, Any],
    kind: str,
    architecture_id: str,
    search_id: str,
    parameter_count: int,
    search_config: Any,
) -> None:
    if arm.get("architecture_id") != architecture_id:
        raise BundleRunError("candidate architecture ID does not match campaign arm")
    if arm.get("search_id") != search_id:
        raise BundleRunError("candidate search ID does not match campaign arm")
    expected_params = arm.get("expected_parameter_count")
    if expected_params is not None and int(expected_params) != int(parameter_count):
        raise BundleRunError(
            f"candidate parameter_count={parameter_count} does not match campaign "
            f"expected_parameter_count={expected_params}"
        )
    expected_kind = (
        "raw_policy"
        if search_id == "raw-policy"
        else "regularized_mcts"
        if search_id == "regularized-policy-mcts"
        else "puct"
        if search_id == "puct"
        else "gumbel"
        if search_id.startswith("gumbel-")
        else None
    )
    if expected_kind is None or kind != expected_kind:
        raise BundleRunError(
            f"search ID {search_id!r} is not bound to candidate kind {kind!r}"
        )
    if search_id == "gumbel-exact-budget" and not bool(
        getattr(search_config, "exact_budget_sh", False)
    ):
        raise BundleRunError("gumbel-exact-budget requires exact_budget_sh=true")
    if search_id == "gumbel-legacy-modal":
        if bool(getattr(search_config, "exact_budget_sh", False)):
            raise BundleRunError("gumbel-legacy-modal requires exact_budget_sh=false")
        if bool(getattr(search_config, "play_sh_winner", False)):
            raise BundleRunError("gumbel-legacy-modal requires play_sh_winner=false")


def _resolved_search_config(kind: str, path: str | Path) -> Any:
    raw = _read_json_object(path, label=f"{kind} search config")
    if kind == "raw_policy":
        if raw:
            raise BundleRunError("raw_policy search config must be an empty JSON object")
        return None
    config_type = {
        "gumbel": GumbelChanceMCTSConfig,
        "puct": RustMCTSConfig,
        "regularized_mcts": RegularizedMCTSConfig,
    }.get(kind)
    if config_type is None:
        raise BundleRunError(f"unsupported operator kind {kind!r}")
    fields = {field.name: field for field in dataclasses.fields(config_type)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise BundleRunError(f"{kind} search config has unknown fields: {', '.join(unknown)}")
    if "colors" in raw:
        raw["colors"] = tuple(str(color) for color in raw["colors"])
    try:
        return config_type(**raw)
    except (TypeError, ValueError) as exc:
        raise BundleRunError(f"invalid {kind} search config: {exc}") from exc


def _operator_and_identity(
    *,
    kind: str,
    architecture_id: str,
    search_id: str,
    checkpoint: str | Path,
    search_config_path: str | Path,
    device: str,
    require_public_information: bool,
    arm_id: str | None = None,
) -> tuple[MeasuredSearchOperator, EntityGraphRustEvaluator, dict[str, Any]]:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise BundleRunError(f"checkpoint does not exist: {checkpoint_path}")
    policy = EntityGraphPolicy.load(checkpoint_path, device=device)
    parameter_count = sum(parameter.numel() for parameter in policy.model.parameters())
    architecture_payload = _jsonable(policy.config)
    search_config = _resolved_search_config(kind, search_config_path)
    search_payload = {"kind": kind, "config": _jsonable(search_config)}
    evaluator = EntityGraphRustEvaluator(
        policy,
        config=EntityGraphRustEvaluatorConfig(
            public_observation=bool(require_public_information),
            cache_size=0,
        ),
    )
    operator = MeasuredSearchOperator(kind, evaluator, config=search_config)
    identity = {
        "arm_id": arm_id,
        "architecture_id": str(architecture_id),
        "parameter_count": int(parameter_count),
        "architecture_config_sha256": canonical_json_sha256(architecture_payload),
        "search_id": str(search_id),
        "search_config_sha256": canonical_json_sha256(search_payload),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    return operator, evaluator, identity


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--budget-regime", choices=("equal_work", "equal_time"), required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--candidate-kind", choices=SEARCH_KINDS, required=True)
    parser.add_argument("--candidate-architecture-id", required=True)
    parser.add_argument("--candidate-search-id", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-search-config", required=True)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--reference-kind", choices=SEARCH_KINDS, required=True)
    parser.add_argument("--reference-architecture-id", required=True)
    parser.add_argument("--reference-search-id", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--reference-search-config", required=True)
    parser.add_argument("--max-decisions", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-authoritative-hidden-state",
        action="store_true",
        help="Opt into a diagnostic that may use authoritative hidden state. Public information is required by default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    require_public = not bool(args.allow_authoritative_hidden_state)
    candidate_evaluator: EntityGraphRustEvaluator | None = None
    reference_evaluator: EntityGraphRustEvaluator | None = None
    try:
        campaign, campaign_arm = _load_campaign_arm(args.campaign, args.arm_id)
        campaign_id = str(campaign["campaign_id"])
        campaign_requires_public = campaign["required_information_regime"] == "public_only"
        if campaign_requires_public and not require_public:
            raise BundleRunError(
                "cannot use --allow-authoritative-hidden-state with a public_only campaign"
            )
        seeds, seed_manifest = load_seed_manifest(args.seed_manifest)
        training_manifest = load_training_manifest(args.training_manifest)
        candidate, candidate_evaluator, candidate_raw = _operator_and_identity(
            kind=args.candidate_kind,
            architecture_id=args.candidate_architecture_id,
            search_id=args.candidate_search_id,
            checkpoint=args.candidate_checkpoint,
            search_config_path=args.candidate_search_config,
            device=args.device,
            require_public_information=require_public,
            arm_id=args.arm_id,
        )
        reference, reference_evaluator, reference_raw = _operator_and_identity(
            kind=args.reference_kind,
            architecture_id=args.reference_architecture_id,
            search_id=args.reference_search_id,
            checkpoint=args.reference_checkpoint,
            search_config_path=args.reference_search_config,
            device=args.device,
            require_public_information=require_public,
        )
        candidate_identity = ArmIdentity(**candidate_raw)
        candidate_search_config = getattr(candidate.search, "config", None)
        _assert_search_arm_binding(
            arm=campaign_arm,
            kind=args.candidate_kind,
            architecture_id=candidate_identity.architecture_id,
            search_id=candidate_identity.search_id,
            parameter_count=candidate_identity.parameter_count,
            search_config=candidate_search_config,
        )
        reference_identity = FrozenReference(
            reference_id=args.reference_id,
            **{key: value for key, value in reference_raw.items() if key != "arm_id"},
        )
        bundle = build_result_bundle(
            candidate,
            reference,
            campaign_id=campaign_id,
            run_id=args.run_id,
            budget_regime=args.budget_regime,
            candidate_identity=candidate_identity,
            reference_identity=reference_identity,
            seeds=seeds,
            seed_manifest=seed_manifest,
            training_manifest=training_manifest,
            code_provenance=capture_git_provenance(args.repo),
            max_decisions=args.max_decisions,
            require_public_information=require_public,
        )
        _write_json_atomic(args.out, bundle)
        print(
            json.dumps(
                {
                    "valid": True,
                    "out": str(Path(args.out).resolve()),
                    "pairs": len(seeds),
                    "games": len(bundle["games"]),
                    "information_regime": bundle["required_information_regime"],
                }
            )
        )
        return 0
    except (BundleRunError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}))
        return 2
    finally:
        for evaluator in (candidate_evaluator, reference_evaluator):
            close = getattr(evaluator, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    raise SystemExit(main())

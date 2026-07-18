#!/usr/bin/env python3
"""Issue an exact V15-direct Stage-C diagnostic learner-parent authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_one_dose_train as one_dose  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--coherent-corpus-admission", required=True, type=Path)
    parser.add_argument("--reviewed-lock-file-sha256", required=True)
    parser.add_argument("--learner-parent-checkpoint", required=True, type=Path)
    parser.add_argument("--learner-parent-training-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        verified = one_dose.verify_training_inputs(
            lock_path=args.lock,
            data_path=args.data,
            validation_path=args.validation_manifest,
            reviewed_lock_file_sha256=args.reviewed_lock_file_sha256,
            coherent_corpus_admission=args.coherent_corpus_admission,
        )
        authority = one_dose.issue_direct_independent_parent_authority(
            verified,
            parent_checkpoint_path=args.learner_parent_checkpoint,
            parent_training_report_path=args.learner_parent_training_report,
            output_path=args.output,
        )
    except (OSError, one_dose.ExecutorError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(authority, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

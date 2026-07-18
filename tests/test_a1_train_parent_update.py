from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from tools import a1_train_parent_update as launch


def _args(tmp_path: Path, *, data: Path) -> argparse.Namespace:
    parent = tmp_path / "parent.pt"
    initializer = tmp_path / "initializer.pt"
    parent.write_bytes(b"exact parent")
    initializer.write_bytes(parent.read_bytes())
    return argparse.Namespace(
        data=str(data),
        parent_checkpoint=str(parent),
        init_checkpoint=str(initializer),
        information_contract_migration_receipt="",
        checkpoint=str(tmp_path / "candidate.pt"),
        report=str(tmp_path / "report.json"),
        python=sys.executable,
        master_port=29500,
    )


def test_direct_launcher_passes_memmap_directory_not_metadata_file(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_meta.json").write_text("{}\n", encoding="utf-8")

    command = launch.command_from_args(_args(tmp_path, data=corpus))

    data_index = command.index("--data") + 1
    assert command[data_index] == str(corpus.resolve())
    assert command[data_index] != str((corpus / "corpus_meta.json").resolve())
    assert command[1:6] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "--master-port=29500",
    ]


def test_direct_launcher_rejects_directory_without_corpus_metadata(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with pytest.raises(SystemExit, match="requires regular corpus_meta.json"):
        launch.command_from_args(_args(tmp_path, data=corpus))

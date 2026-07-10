from __future__ import annotations

import sys
from pathlib import Path

from tools import build_memmap_corpus as builder


def test_source_list_reaches_builder_without_argv_expansion(
    tmp_path: Path, monkeypatch
) -> None:
    source_list = tmp_path / "sources.txt"
    source_list.write_text("/harvest/worker_000\n/harvest/worker_001\n")
    captured = {}

    def fake_build(sources, out, **kwargs):
        captured.update(sources=sources, out=out, kwargs=kwargs)

    monkeypatch.setattr(builder, "build_memmap_corpus", fake_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_memmap_corpus.py",
            "--source-list",
            str(source_list),
            "--out",
            str(tmp_path / "corpus"),
        ],
    )

    builder.main()

    assert captured["sources"] == [
        Path("/harvest/worker_000"),
        Path("/harvest/worker_001"),
    ]
    assert captured["out"] == tmp_path / "corpus"

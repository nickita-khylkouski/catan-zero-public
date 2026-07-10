from __future__ import annotations

from tools.train_bc import build_parser


def test_topology_v2_cli_is_explicit_and_defaults_to_no_behavior_change() -> None:
    parser = build_parser()
    defaults = parser.parse_args(
        ["--data", "data", "--checkpoint", "checkpoint.pt", "--report", "report.json"]
    )
    assert defaults.topology_adapter_layers == ""
    assert defaults.topology_adapter_kind == "basis_mean_v1"
    assert defaults.topology_adapter_heads == 4
    assert defaults.topology_adapter_share_weights is False
    assert defaults.topology_adapter_edge_control == "true_topology"

    configured = parser.parse_args(
        [
            "--data",
            "data",
            "--checkpoint",
            "checkpoint.pt",
            "--report",
            "report.json",
            "--topology-adapter-layers",
            "2,4",
            "--topology-adapter-kind",
            "local_attention_v2",
            "--topology-adapter-width",
            "192",
            "--topology-adapter-heads",
            "4",
            "--topology-adapter-share-weights",
            "--topology-adapter-edge-control",
            "self_message",
        ]
    )
    assert configured.topology_adapter_layers == "2,4"
    assert configured.topology_adapter_kind == "local_attention_v2"
    assert configured.topology_adapter_width == 192
    assert configured.topology_adapter_heads == 4
    assert configured.topology_adapter_share_weights is True
    assert configured.topology_adapter_edge_control == "self_message"

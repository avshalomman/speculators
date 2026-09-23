"""The render subcommand serves without hidden-state extraction, render tuning kept."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import launch_vllm  # type: ignore[import-not-found]


def test_render_subcommand_serves_without_extraction(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch_vllm.py", "render", "verifier", "--dry-run", "--", "--port", "8000"],
    )

    launch_vllm.main()

    command = capsys.readouterr().out
    assert "serve verifier" in command
    assert "--port 8000" in command
    assert "--api-server-count" in command
    assert "--speculative_config" not in command
    assert "--kv_transfer_config" not in command

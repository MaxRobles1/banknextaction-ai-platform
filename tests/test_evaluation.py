"""Model selection rules and receipt integrity."""

from __future__ import annotations

import json
from pathlib import Path

from banknextaction.evaluation.receipt import artifact_entry, verify_receipt
from banknextaction.evaluation.selection import Candidate, select_served_model


def _candidate(name: str, f1: float, pr: float, latency: float = 1.0, size: int = 10) -> Candidate:
    return Candidate(name, f1, pr, latency, size)


def test_selection_prefers_validation_f1() -> None:
    selection = select_served_model([_candidate("a", 0.30, 0.20), _candidate("b", 0.28, 0.40)])

    assert selection.served == "a"
    assert "F1" in selection.reason


def test_selection_breaks_f1_ties_with_pr_auc_then_latency() -> None:
    by_pr = select_served_model([_candidate("a", 0.3001, 0.20), _candidate("b", 0.3003, 0.25)])
    by_latency = select_served_model(
        [_candidate("a", 0.30, 0.25, latency=2.0), _candidate("b", 0.30, 0.25, latency=1.0)]
    )

    assert by_pr.served == "b"
    assert by_latency.served == "b"
    assert "latency" in by_latency.reason


def test_verify_receipt_detects_modified_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "modelo.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"weights")
    receipt = {"models": {"m": {"artifacts": [artifact_entry(artifact, tmp_path)]}}}
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))

    assert verify_receipt(receipt_path, tmp_path) == []
    artifact.write_bytes(b"tampered")
    assert verify_receipt(receipt_path, tmp_path) == ["models/modelo.bin"]

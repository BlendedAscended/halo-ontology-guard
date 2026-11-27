"""Deterministic RLVR verifier and baseline evaluator for OntologyGuard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

BOXED_RE = re.compile(r"\\boxed\{(.+)\}\s*$", re.DOTALL)


def extract_boxed_json(text: str) -> dict[str, Any] | None:
    match = BOXED_RE.search(text.strip())
    candidate = match.group(1) if match else text.strip()
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def score_completion(task_type: str, completion: str, ground_truth: dict[str, Any]) -> float:
    parsed = extract_boxed_json(completion)
    if parsed is None:
        return 0.0

    if task_type == "jev_triage_hitl":
        if ground_truth.get("requires_hitl") is True and parsed.get("requires_hitl") is not True:
            return 0.0
        keys = ("vertical_id", "role_id", "intent", "trust_zone", "requires_hitl")
        hits = sum(1 for k in keys if parsed.get(k) == ground_truth.get(k))
        return round(hits / len(keys), 4)

    if task_type == "ontology_command_compiler":
        if ground_truth.get("decision") == "reject_illegal_transition":
            if parsed.get("decision") != "reject_illegal_transition":
                return 0.0
            return 1.0 if parsed.get("entity_table") == ground_truth.get("entity_table") else 0.5
        keys = ("decision", "vertical_id", "entity_table", "from_state", "to_state")
        hits = sum(1 for k in keys if parsed.get(k) == ground_truth.get(k))
        if parsed.get("requires_human_approval") != ground_truth.get("requires_human_approval"):
            return round(max(0.0, (hits / len(keys)) - 0.4), 4)
        return round(hits / len(keys), 4)

    if task_type == "opposite_family_verifier":
        if parsed.get("verdict") != ground_truth.get("verdict"):
            return 0.0
        expected_v = set(ground_truth.get("violations", []))
        got_v = set(parsed.get("violations", [])) if isinstance(parsed.get("violations"), list) else set()
        if not expected_v and not got_v:
            return 1.0
        if not expected_v or not got_v:
            return 0.5
        return round(0.5 + 0.5 * (len(expected_v & got_v) / len(expected_v | got_v)), 4)

    raise ValueError(f"Unknown task_type: {task_type}")


def verify_dataset_oracle(dataset_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = [
        score_completion(r["task_type"], f"\\boxed{{{r['answer']}}}", json.loads(r["answer"]))
        for r in rows
    ]
    return {
        "rows": len(rows),
        "oracle_mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify OntologyGuard RLVR dataset against deterministic oracle.")
    ap.add_argument("--dataset", type=Path, default=Path("data/ontology_guard/grpo_online.jsonl"))
    args = ap.parse_args()
    print(json.dumps(verify_dataset_oracle(args.dataset), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

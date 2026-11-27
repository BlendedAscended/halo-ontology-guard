"""Convert generated synthetic ontology JSONL files into Halo SFT and Online GRPO format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(input_dir: Path, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_all: list[str] = []
    sft_compiler: list[str] = []
    grpo_online: list[str] = []

    for fname in ("jev_triage_hitl.jsonl", "ontology_command_compiler.jsonl", "opposite_family_verifier.jsonl"):
        src = input_dir / fname
        if not src.is_file():
            raise FileNotFoundError(f"Missing dataset file: {src}")
        for line in src.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            canonical_json = json.dumps(row["ground_truth"], sort_keys=True, separators=(",", ":"))
            boxed_answer = f"\\boxed{{{canonical_json}}}"

            sft_msgs = [
                row["messages"][0],
                row["messages"][1],
                {"role": "assistant", "content": boxed_answer},
            ]
            sft_line = json.dumps({"messages": sft_msgs, "task_type": row["task_type"]}, ensure_ascii=False)
            sft_all.append(sft_line)
            if row["task_type"] == "ontology_command_compiler":
                sft_compiler.append(sft_line)

            grpo_line = json.dumps(
                {
                    "prompt": row["prompt"],
                    "answer": canonical_json,
                    "task_type": row["task_type"],
                },
                ensure_ascii=False,
            )
            grpo_online.append(grpo_line)

    (out_dir / "sft_all.jsonl").write_text("\n".join(sft_all) + "\n", encoding="utf-8")
    (out_dir / "sft_ontology_compiler.jsonl").write_text("\n".join(sft_compiler) + "\n", encoding="utf-8")
    (out_dir / "grpo_online.jsonl").write_text("\n".join(grpo_online) + "\n", encoding="utf-8")
    return {
        "sft_all": len(sft_all),
        "sft_ontology_compiler": len(sft_compiler),
        "grpo_online": len(grpo_online),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    counts = convert(args.input_dir, args.out_dir)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

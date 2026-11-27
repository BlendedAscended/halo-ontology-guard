"""Convert generated synthetic ontology JSONL splits into Halo SFT and Online GRPO files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_rows(src_file: Path) -> tuple[list[str], list[str], list[str]]:
    sft_all: list[str] = []
    sft_compiler: list[str] = []
    grpo_rows: list[str] = []
    for line in src_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        canonical_json = json.dumps(row["ground_truth"], sort_keys=True, separators=(",", ":"))
        boxed_answer = f"\\boxed{{{canonical_json}}}"

        sft_msgs = [
            row["messages"][0],
            row["messages"][1],
            {"role": "assistant", "content": boxed_answer},
        ]
        sft_line = json.dumps(
            {"id": row["id"], "messages": sft_msgs, "task_type": row["task_type"]},
            ensure_ascii=False,
        )
        sft_all.append(sft_line)
        if row["task_type"] == "ontology_command_compiler":
            sft_compiler.append(sft_line)

        grpo_rows.append(
            json.dumps(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "answer": canonical_json,
                    "task_type": row["task_type"],
                },
                ensure_ascii=False,
            )
        )
    return sft_all, sft_compiler, grpo_rows


def convert(input_dir: Path, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_src = input_dir / "train_all.jsonl"
    eval_src = input_dir / "eval_holdout.jsonl"
    if not train_src.is_file() or not eval_src.is_file():
        raise FileNotFoundError(f"Expected train_all.jsonl and eval_holdout.jsonl in {input_dir}")

    sft_train, sft_comp_train, grpo_train = _format_rows(train_src)
    sft_eval, _, grpo_eval = _format_rows(eval_src)

    (out_dir / "sft_all.jsonl").write_text("\n".join(sft_train) + "\n", encoding="utf-8")
    (out_dir / "sft_ontology_compiler.jsonl").write_text("\n".join(sft_comp_train) + "\n", encoding="utf-8")
    (out_dir / "grpo_online.jsonl").write_text("\n".join(grpo_train) + "\n", encoding="utf-8")
    (out_dir / "sft_eval.jsonl").write_text("\n".join(sft_eval) + "\n", encoding="utf-8")
    (out_dir / "grpo_eval.jsonl").write_text("\n".join(grpo_eval) + "\n", encoding="utf-8")

    return {
        "sft_train": len(sft_train),
        "sft_ontology_compiler_train": len(sft_comp_train),
        "grpo_online_train": len(grpo_train),
        "eval_holdout": len(grpo_eval),
        "total": len(grpo_train) + len(grpo_eval),
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

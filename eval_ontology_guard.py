"""Deterministic RLVR verifier and live OpenRouter / Vertex / vLLM benchmark runner."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request

BOXED_RE = re.compile(r"\\boxed\{(.+)\}\s*$", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_boxed_json(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    candidates = []
    match = BOXED_RE.search(stripped)
    if match:
        candidates.append(match.group(1).strip())
    fence = FENCE_RE.search(stripped)
    if fence:
        candidates.append(fence.group(1).strip())
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if 0 <= first_brace < last_brace:
        candidates.append(stripped[first_brace : last_brace + 1])
    candidates.append(stripped)

    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


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

    if task_type == "voice_front_desk":
        if ground_truth.get("requires_hitl") is True and parsed.get("requires_hitl") is not True:
            return 0.0
        keys = ("trade", "vertical_id", "caller_intent", "urgency", "action", "slot_captured", "requires_hitl")
        hits = sum(1 for k in keys if parsed.get(k) == ground_truth.get(k))
        return round(hits / len(keys), 4)

    if task_type == "forge_treg_planner":
        if ground_truth.get("refuse_unverified_metrics") is True and parsed.get("refuse_unverified_metrics") is not True:
            return 0.0
        if ground_truth.get("requires_baa_gate") is True and parsed.get("requires_baa_gate") is not True:
            return 0.0
        keys = ("vertical_id", "cohort", "selected_modules", "treg_families", "refuse_unverified_metrics", "requires_baa_gate")
        hits = sum(1 for k in keys if parsed.get(k) == ground_truth.get(k))
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
    by_task: dict[str, int] = {}
    for r in rows:
        by_task[r["task_type"]] = by_task.get(r["task_type"], 0) + 1
    return {
        "rows": len(rows),
        "by_task": by_task,
        "oracle_mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
    }


def _resolve_api_credentials(backend: str, env_file: Path | None = None) -> tuple[str, str, dict[str, str]]:
    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key and env_file and env_file.is_file():
            for ln in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ln.startswith("OPENROUTER_API_KEY="):
                    key = ln.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment or --env-file")
        return "https://openrouter.ai/api/v1/chat/completions", key, {}

    if backend == "vllm":
        base = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1/chat/completions")
        key = os.environ.get("VLLM_API_KEY", "EMPTY")
        return base, key, {}

    if backend == "vertex":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global").strip()
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT required for --backend vertex")
        tok = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True, timeout=10).strip()
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        url = f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions"
        return url, tok, {"x-goog-user-project": project}

    raise ValueError(f"Unsupported backend: {backend}")


def _call_chat_completion(
    url: str,
    token: str,
    model: str,
    messages: list[dict[str, str]],
    ground_truth: dict[str, Any],
    timeout_s: float = 60.0,
    max_tokens: int = 1536,
    extra_headers: dict[str, str] | None = None,
) -> tuple[str, float, dict[str, Any]]:
    schema_hint = (
        f"\nRequired exact JSON keys: {sorted(ground_truth.keys())}. "
        "Output ONLY canonical JSON wrapped in \\boxed{...} with no extra prose."
    )
    augmented = [
        {"role": messages[0]["role"], "content": messages[0]["content"] + schema_hint},
        messages[1],
    ]
    payload = {
        "model": model,
        "messages": augmented,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    usage = body.get("usage") or {}
    return str(content), latency_ms, usage


def select_stratified_sample(dataset_path: Path, limit_per_task: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        buckets.setdefault(row["task_type"], []).append(row)
    selected: list[dict[str, Any]] = []
    for task in sorted(buckets):
        selected.extend(buckets[task][:limit_per_task])
    return selected


def run_live_benchmark(
    *,
    dataset_path: Path,
    backend: str,
    models: list[str],
    limit_per_task: int = 6,
    concurrency: int = 8,
    max_tokens: int = 1536,
    env_file: Path | None = None,
    call_fn=None,
) -> dict[str, Any]:
    url, token, extra_headers = (
        ("mock://local", "mock", {}) if call_fn is not None else _resolve_api_credentials(backend, env_file)
    )
    caller = call_fn or (
        lambda u, t, m, msgs, gt: _call_chat_completion(
            u, t, m, msgs, gt, max_tokens=max_tokens, extra_headers=extra_headers
        )
    )
    sample = select_stratified_sample(dataset_path, limit_per_task=limit_per_task)

    def _eval_model(model: str) -> dict[str, Any]:
        per_task_scores: dict[str, list[float]] = {}
        latencies: list[float] = []
        boxed_hits = 0
        errors = 0

        def _eval_one(row: dict[str, Any]) -> tuple[str, float, float, bool, bool]:
            gt = json.loads(row["answer"])
            try:
                text, lat, _ = caller(url, token, model, row["prompt"], gt)
                sc = score_completion(row["task_type"], text, gt)
                has_box = bool(BOXED_RE.search(text.strip()))
                return row["task_type"], sc, lat, has_box, False
            except Exception:
                return row["task_type"], 0.0, 0.0, False, True

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futs = [pool.submit(_eval_one, r) for r in sample]
            for fut in as_completed(futs):
                ttype, sc, lat, has_box, erred = fut.result()
                per_task_scores.setdefault(ttype, []).append(sc)
                if lat > 0:
                    latencies.append(lat)
                if has_box:
                    boxed_hits += 1
                if erred:
                    errors += 1

        all_scores = [s for vals in per_task_scores.values() for s in vals]
        lat_sorted = sorted(latencies)
        p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
        p95 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))] if lat_sorted else 0.0

        return {
            "model": model,
            "backend": backend,
            "evaluated_rows": len(sample),
            "errors": errors,
            "mean_reward": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
            "boxed_format_rate": round(boxed_hits / len(sample), 4) if sample else 0.0,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "task_means": {
                k: round(sum(v) / len(v), 4) for k, v in sorted(per_task_scores.items())
            },
        }

    with ThreadPoolExecutor(max_workers=min(len(models), 6) or 1) as model_pool:
        model_reports = list(model_pool.map(_eval_model, models))
    model_reports.sort(key=lambda r: r["mean_reward"], reverse=True)

    return {
        "dataset": str(dataset_path),
        "limit_per_task": limit_per_task,
        "total_sample_rows": len(sample),
        "models": model_reports,
    }


def render_markdown_table(report: dict[str, Any]) -> str:
    lines = [
        "# OntologyGuard Baseline Evaluation Report",
        "",
        f"Evaluated on `{report['dataset']}` ({report['total_sample_rows']} stratified holdout rows across 5 tasks).",
        "",
        "| Model | Mean RLVR Reward | `\\boxed{}` Rate | JEV HITL | Ontology | Voice Desk | Treg Plan | G14 Verifier | P50 Latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in report["models"]:
        tm = m["task_means"]
        lines.append(
            f"| `{m['model']}` | **{m['mean_reward']:.3f}** | {m['boxed_format_rate']:.2f} | "
            f"{tm.get('jev_triage_hitl', 0.0):.2f} | {tm.get('ontology_command_compiler', 0.0):.2f} | "
            f"{tm.get('voice_front_desk', 0.0):.2f} | {tm.get('forge_treg_planner', 0.0):.2f} | "
            f"{tm.get('opposite_family_verifier', 0.0):.2f} | {m['latency_p50_ms']:.0f} ms |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify OntologyGuard RLVR dataset or benchmark live endpoints.")
    ap.add_argument("--mode", choices=("oracle", "live"), default="oracle")
    ap.add_argument("--dataset", type=Path, default=Path("data/ontology_guard/grpo_eval.jsonl"))
    ap.add_argument("--backend", choices=("openrouter", "vllm", "vertex"), default="openrouter")
    ap.add_argument("--models", default="google/gemma-4-26b-a4b-it,thinkingmachines/inkling-2-mini,qwen/qwen3.8-flash")
    ap.add_argument("--limit-per-task", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.mode == "oracle":
        res = verify_dataset_oracle(args.dataset)
        print(json.dumps(res, indent=2))
        return 0

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    report = run_live_benchmark(
        dataset_path=args.dataset,
        backend=args.backend,
        models=model_list,
        limit_per_task=args.limit_per_task,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        env_file=args.env_file,
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown_table(report), encoding="utf-8")
    print(render_markdown_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

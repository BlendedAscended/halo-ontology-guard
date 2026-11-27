# OntologyGuard Baseline Evaluation Report

Evaluated on `data/ontology_guard/grpo_eval.jsonl` (25 stratified holdout rows across 5 tasks).

| Model | Mean RLVR Reward | `\boxed{}` Rate | JEV HITL | Ontology | Voice Desk | Treg Plan | G14 Verifier | P50 Latency |
|---|---|---|---|---|---|---|---|---|
| `z-ai/glm-5.3-flash` | **0.123** | 0.16 | 0.12 | 0.16 | 0.00 | 0.33 | 0.00 | 2824 ms |
| `deepseek/deepseek-v4.1-flash` | **0.000** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2663 ms |
| `google/gemini-3.1-flash-lite` | **0.506** | 1.00 | 0.52 | 0.80 | 0.34 | 0.87 | 0.00 | 902 ms |

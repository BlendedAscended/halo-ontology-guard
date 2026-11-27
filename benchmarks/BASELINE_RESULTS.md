# OntologyGuard Baseline Evaluation Report

Evaluated on `data/ontology_guard/grpo_eval.jsonl` (30 stratified holdout rows across 5 tasks).

| Model | Mean RLVR Reward | `\boxed{}` Rate | JEV HITL | Ontology | Voice Desk | Treg Plan | G14 Verifier | P50 Latency |
|---|---|---|---|---|---|---|---|---|
| `openai/gpt-6.2-sol` | **0.536** | 1.00 | 0.50 | 0.60 | 0.36 | 0.89 | 0.33 | 2546 ms |
| `upstage/solar-mini-4-5` | **0.490** | 0.60 | 0.47 | 0.53 | 0.31 | 0.81 | 0.33 | 742 ms |
| `google/gemma-4-26b-a4b-it` | **0.483** | 1.00 | 0.57 | 0.53 | 0.29 | 0.69 | 0.33 | 876 ms |
| `anthropic/claude-haiku-5.5` | **0.473** | 0.93 | 0.50 | 0.53 | 0.33 | 0.83 | 0.17 | 2258 ms |
| `nvidia/nemotron-3.5-super-vl` | **0.390** | 0.70 | 0.10 | 0.53 | 0.26 | 0.89 | 0.17 | 4267 ms |
| `deepseek/deepseek-v4.1-flash-1017` | **0.378** | 0.67 | 0.33 | 0.00 | 0.33 | 0.89 | 0.33 | 6698 ms |
| `google/gemini-4-hydrogen` | **0.341** | 0.57 | 0.10 | 0.13 | 0.17 | 0.72 | 0.58 | 7864 ms |
| `meta/muse-ember` | **0.322** | 0.47 | 0.10 | 0.13 | 0.07 | 0.89 | 0.42 | 4407 ms |
| `qwen/qwen3.8-flash` | **0.280** | 0.43 | 0.17 | 0.13 | 0.24 | 0.69 | 0.17 | 16077 ms |
| `inclusionai/lingsante-3.1-air-vl` | **0.274** | 0.63 | 0.23 | 0.20 | 0.05 | 0.56 | 0.33 | 11193 ms |
| `thinkingmachines/inkling-2-small` | **0.214** | 0.37 | 0.10 | 0.00 | 0.17 | 0.31 | 0.50 | 6378 ms |
| `thinkingmachines/inkling-2-mini` | **0.069** | 0.27 | 0.00 | 0.00 | 0.10 | 0.17 | 0.08 | 7705 ms |

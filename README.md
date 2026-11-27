# OntologyGuard: Deterministic State-Machine RLVR on Halo

Open-source post-training benchmark for **schema-valid business workflow compilation**, **human-in-the-loop (HITL) escalation precision**, and **receipt invariant auditing** across 24 industry schemas using **Halo** (`SFT` -> `Online GRPO` with rule accuracy + White Circle policy rewards).

100% synthetic, zero customer data, zero PII/PHI.

## Tracks Targeted
- **Google DeepMind:** Best Gemma post-training (`google/gemma-4-26B-A4B-it`)
- **White Circle:** Best use of White Circle (policy-scored GRPO trajectories)
- **OpenRouter / Modal / Baseten / AWS:** Reproducible SFT -> Online GRPO -> before/after checkpoint evaluation pipeline

## Quickstart (Blackwell Docker)

```bash
# 0. Verify included synthetic datasets and deterministic oracle locally
python3 eval_ontology_guard.py --dataset data/ontology_guard/grpo_online.jsonl
pytest -q

docker pull public.ecr.aws/whitecircle/halo:blackwell

# 1. Baseline SFT warmup (Gemma-4 26B-A4B MoE)
torchrun --nproc_per_node=8 scripts/training/sft.py configs/sft-gemma4-26b-a4b-ontology.yaml

# 2. Create White Circle safety policies (optional for White Circle track)
WHITECIRCLE_API_KEY=wc-... halo run create-whitecircle-policies policies.json

# 3. Launch vLLM rollout server (GPU 0) + Online GRPO trainer (GPUs 1-2)
VLLM_MODEL=checkpoints/sft-gemma4-26b-a4b-ontology \
  docker compose -f docker-compose.vllm.yml up -d vllm-server

WHITECIRCLE_API_KEY=wc-... CUDA_VISIBLE_DEVICES=1,2 \
  halo launch grpo/online configs/online-grpo-gemma4-26b-a4b-whitecircle.yaml -n 2
```

## Deterministic Verifiable Rewards
1. **Zero-miss HITL escalation:** Requests asking for a human operator must emit `requires_hitl: true`.
2. **State-machine edge enforcement:** Legal lifecycle transitions only; `drafted -> approved` bypasses must return `reject_illegal_transition`.
3. **High-impact approval & receipt conservation:** Financial/legal actions require `requires_human_approval: true`; foreign scope or `NaN` cost receipts must fail closed.

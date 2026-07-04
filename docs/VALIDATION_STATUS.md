# Validation Status — medgemma-27b-med-kfp-ft-eval-pipeline

**Model:** `google/medgemma-27b-it`
**Task:** Medical MCQ fine-tuning on AIIMS/NEET PG entrance exam questions (openlifescienceai/medmcqa)
**Platform:** Kubeflow Pipelines on NVIDIA DGX Spark (GB10 Blackwell, 128 GB unified memory)
**Last updated:** 2026-07-04

---

## Current Status

| Component              | Status                                         |
| ---------------------- | ---------------------------------------------- |
| `baseline_eval`        | ✅ run-001 — 0.7550 accuracy                    |
| `baseline_safety_eval` | ✅ run-001 — 4.95 avg safety score              |
| `fine_tune`            | ✅ run-001 — 1.1594 train loss, epoch 0.1114/3.0 (5h budget) |
| `post_finetune_eval`   | ✅ run-001 — 0.7550 accuracy (no change)        |
| `safety_eval`          | ✅ run-001 — 4.96 avg safety score              |
| `deployment_gate`      | ✅ run-001 — PASS (degenerate)                  |

**run-001 complete. All six pipeline stages executed successfully. Gate: PASS.**

**Key finding:** MedGemma-27B-IT baseline is already 75.5% on MedMCQA — it is a
medical-domain specialist. Fine-tuning on the same distribution with only 0.11 epochs
(5h budget at batch=1 for 27B) produced zero accuracy gain. Run-002 needs ≥24h budget.

---

## Run Table

| Run     | Purpose           | Result          | Baseline Acc | Post-FT Acc | ΔAcc  | Key Finding |
| ------- | ----------------- | --------------- | ------------ | ----------- | ----- | ----------- |
| run-001 | Establish baseline + first fine-tune | PASS (degenerate) | 0.755 | 0.755 | 0.000 | Medical specialist baseline too high; 0.11 epochs insufficient for adaptation |

---

## What Is Implemented

### Infrastructure (inherited from platform template)
- KFP v2 pipeline scaffold with all 8 stages wired
- MLflow run-per-stage tracking (one MLflow run per stage, experiment = project name)
- `purge_kfp_mlflow.py` (never run automatically — explicit user command only)
- Nsight Operator integration — add `kubernetes.add_pod_label(task, "nvidia-nsight-profile", "enabled")` to profile any stage
- BF16 direct loading with `max_memory={0: "100GiB"}` (Blackwell GB10 unified memory budget)
- Time-budgeted training: `target_hours=6.0` with `overhead_hours=1.5` → ~4.5h effective training window

### Project-specific
- `config.yaml` — google/medgemma-27b-it, medmcqa, LoRA r=16/α=32, 7-module target, batch=1/grad_accum=8, 5h budget, Phi-4 judge
- `formatters.py` — `format_medmcqa()`: cop-index (0–3) → A/B/C/D letter, same schema as qwen25-7b-medmcqa project
- `loaders.py` — `medmcqa` lambda: `load_dataset` train split → formatter map → empty-instruction filter
- `eval_helpers.py` — `extract_answer()` covering MCQ letter (a-e), `make_infer_fn()` via `apply_chat_template`
- `notebook.ipynb` — all 7 user code blocks implemented and validated on run-001

---

## What Is Still Pending

- run-002: extend training budget to ≥24h (≥0.5 epochs on 27B at batch=1) — or reconsider experiment design (see Known Issues)
- Investigate whether fine-tuning a medical specialist on its own training distribution provides any signal at all

---

## Known Issues

### run-001: Zero accuracy gain — degenerate PASS

**Root cause:** Two compounding factors:
1. MedGemma-27B-IT is a medical-domain specialist pre-trained on medical corpora; its zero-shot MedMCQA accuracy (75.5%) is already near the practical ceiling for this model size.
2. At batch=1 (required for 27B within 100 GiB), the 5h training budget covered only 0.1114 epochs (~2,033 effective steps). This is insufficient to shift the adapter weights meaningfully.

**Implication:** A 24h budget would cover ~0.5 epochs. Whether that produces improvement depends on whether the model has headroom above 75.5% — unclear given it's already domain-specialized.

**Alternative experiment:** Consider a harder evaluation set (e.g. USMLE Step 2/3 or MedQA) where the model's baseline is lower, making fine-tuning signal visible.

> **Platform-level fixes** (bitsandbytes on Blackwell, trl 0.29 API, PIP_CONSTRAINT, nsys mmap, CUPTI privileges) are already incorporated in this template. See [qwen25-7b-arc-ft-eval-pipeline/docs/VALIDATION_STATUS.md](https://github.com/miramar-labs-org/qwen25-7b-arc-ft-eval-pipeline/blob/main/docs/VALIDATION_STATUS.md) for the full fix history (first green run).

---

## Fixed Issues

None specific to this project. All platform fixes were inherited pre-baked from the template.

---

## Latest Nsight Finding

No profiling runs yet.

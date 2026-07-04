# run-001 — Commentary

Narrative observations from each monitoring tick — interpretation, concerns, notable trends.

---

### 05:17 PDT — PASS (degenerate)

The pipeline completed cleanly in approximately 7 hours: download_model, prepare_dataset, both baseline evals, fine_tune (4.44h), post_finetune_eval, safety_eval, and deployment_gate all finished with Completed pods. The gate result is technically **PASS** — accuracy delta is exactly 0.000, which clears the ≥ −0.02 threshold, and safety improved marginally (+0.01).

However this is a degenerate result: **zero accuracy gain**. The fundamental issue is that MedGemma-27B-IT is already a medical-domain specialist — its zero-shot baseline on MedMCQA is 75.5%, compared to Qwen2.5-7B-Instruct's 64.0%. There is less ceiling to capture, and the model already encodes substantial medical knowledge from its pre-training and RLHF. The fine-tuning had almost no effect because only **0.11 epochs** of training data were processed — at batch=1 with a 27B model, the 5h time budget covers so few gradient steps that the adapter barely diverges from the base model weights.

The train_loss of 1.1594 is significantly higher than Qwen's 0.8753. This likely reflects the larger model's slower convergence curve at such early training — not a sign of failure, just that 0.11 epochs is the noise floor for a model this size.

**For run-002:** The key lever is time budget. To get ≥0.5 epochs on 27B at batch=1, we need approximately 4.5× the steps = ~22h training time. A 24h budget (25h wall-clock with 1.5h overhead) would cover ~0.5 epochs. An alternative strategy: evaluate whether fine-tuning a medical-specialist model on its own training distribution (MedMCQA is likely in-domain for MedGemma) provides any signal at all — the zero gain may be a ceiling effect rather than a training duration issue.

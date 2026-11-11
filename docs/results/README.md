# Evaluation results

| File | What |
|---|---|
| `offline_synthetic.*` | Offline mode (deterministic stand-ins): retrieval on both gold sets + synthetic-image ablations. Harness check. |
| `offline_standin_iu_xray.*` | Offline heuristic "VLM" on 12 real IU X-rays: the baseline that real mode must beat. |
| `real_mode_cpu_iu_xray.*` | **Real mode pilot**: BGE-M3, bge-reranker-v2-m3, Qwen2.5-VL-3B (CPU, built-in server), NLI verifier v2, 12 real IU X-ray studies, all four variants. |

Reproduce: see docs/EVALUATION.md. Per-case outputs (which quote the radiologists' reports) and the clinician review
sheet are written to `eval_results/` locally and are not committed.

## Reading the real-mode pilot

* **Grounding (RQ1, RQ4)**: ungrounded-claim rate full 0.28 < single-pass 0.38 < no retrieval 1.00: retrieval grounds
  the drafts and the self-verification loop removes further unsupported claims.
* **Safety behaviour**: 10 of 12 full-system drafts ended as `needs_review` - the agent declines to vouch for drafts it
  cannot ground; the release gate (ungrounded <= 0.10) correctly fails this configuration.
* **Vision is the bottleneck**: finding F1 0.11; the 3B model at 256 image patches over-calls effusion/cardiomegaly on
  normal films (and produced identical drafts for different films). Text-only predicts "normal" for everything, which
  inflates its normal/abnormal accuracy on a mostly-normal sample; F1 on abnormal findings is the primary metric.
* **Report similarity** with the biomedical sentence encoder sits near 0.92 for any pair of chest X-ray texts and
  should only be used to compare variants, not as an absolute quality score.
* **Grounded is not the same as relevant**: an off-topic but faithfully cited sentence passes verification;
  clinician review (the review sheet) covers relevance.
* Pilot only: 12 studies, rule-based labels, CPU. The thesis run should use Qwen2.5-VL-7B on GPU and all 150 studies.

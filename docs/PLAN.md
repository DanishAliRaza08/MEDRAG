# Project plan and traceability

## Research questions → objectives → implementation → evidence

| RQ | Objective | Implementation | How it is measured / verified |
|---|---|---|---|
| RQ1: does hybrid RAG reduce ungrounded statements? | O1: dense + sparse + re-ranking pipeline | `retrieval/pipeline.py` | Recall@K, MRR, latency per strategy; `full` vs `no_retrieval` ungrounded-claim rate |
| RQ2: multimodal retrieval + VLM for image-based reasoning | O2: VLM + similar-case retrieval, ≥ 2 modalities | `imaging/`, `agent/graph.py:inspect_image` (X-ray, fundus, histopathology) | `full` vs `text_only` finding accuracy and similarity |
| RQ3: privacy + mandatory oversight without degrading performance | O3: de-identification + enforced sign-off | `deid/`, `reports/lifecycle.py`, `service.py` | Tests and audit records (see ARCHITECTURE.md). Diagnostic content preserved after de-identification (`test_deid_preserves_diagnostic_content`) |
| RQ4: does self-verification improve grounding? | O4: bounded verify → re-retrieve loop | `agent/verifier.py`, `agent/graph.py:verify` | `full` vs `single_pass` (real mode); loop behaviour tests |

## Phases

| # | Activity | Status |
|---|---|---|
| 1–2 | Literature review and analysis of core papers | Synopsis |
| 3–4 | Proposal submission and presentation | Synopsis |
| 5 | Dataset selection; start credentialing (PhysioNet / MIMIC-CXR) early | **To do (start now)** |
| 6 | Ingestion and system build | **Done in offline mode**: complete system, 117 tests, CI |
| 6b | Real-mode bring-up on GPU: vLLM + Qwen2.5-VL, BGE-M3, reranker, BiomedCLIP, Qdrant | Code paths written; needs a GPU host and validation |
| 6c | Replace the sample KB with licensed guidelines; open data (ChestX-ray14, OpenI, APTOS/IDRiD fundus, BreakHis/PCam) | To do |
| 7 | Gold sets, experiments on all ablations, expert review of a sample of drafts | Harness ready |
| 8–11 | Thesis presentation, write-up, editing, submission | Write-up in parallel |

## Known limitations / next engineering steps

* The real-mode prompts and JSON parsing (`LLMGenerator`, `QwenVLM`) still need tuning against the served model.
* Verification is lexical (term overlap with cited passages). A natural-language-inference or LLM-judge verifier
  would be a stronger critic for paraphrased output. It would plug into `Verifier.check`.
* DICOM input is supported through optional `pydicom`. Full DICOM tag-level de-identification profiles (PS3.15)
  are future work.
* Single-node SQLite is fine for research. A hospital deployment would use PostgreSQL and MinIO behind the same
  interfaces.

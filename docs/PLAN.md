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
| 5 | Dataset selection; credentialing for MIMIC-CXR (PhysioNet) | Open data done (IU X-ray); **MIMIC-CXR credentialing: student action** |
| 6 | Ingestion and system build | **Done**: complete system, CI, offline + real mode |
| 6b | Real-mode bring-up: Qwen2.5-VL, BGE-M3, bge-reranker-v2-m3, BiomedCLIP, NLI verifier | **Done and validated on CPU** (built-in `medrag llm-server`); vLLM/GPU path ready in docker-compose |
| 6c | Real knowledge base and real images | **Done**: 31 curated CC BY PubMed Central reviews/guidelines; PDF/TXT importer for licensed NICE/WHO/local guidelines; 150 real IU X-ray studies with radiologist reports |
| 7 | Gold sets, experiments on all ablations, expert review of a sample of drafts | Gold sets built (sample, PMC, IU X-ray labels); real-mode pilot run done; clinician review sheet generated; **full-scale GPU run + clinician ratings: next** |
| 8–11 | Thesis presentation, write-up, editing, submission | Write-up in parallel |

## What only the student / supervisor can do

1. **MIMIC-CXR access**: create a PhysioNet account, complete the CITI "Data or Specimens Only Research" course,
   sign the data-use agreement. Then download MIMIC-CXR-JPG and run the same evaluation on it.
2. **GPU time** for the thesis-scale run (e.g. a rented 24 GB GPU for a few hours): `docker compose --profile gpu up`,
   then `medrag eval --real-xray var/datasets/iu_xray --n 150`.
3. **Clinician review**: give `eval_results/clinician_review_sheet.csv` (plus the images) to one or two clinicians,
   ideally blinded to the system variant.
4. **Licensed guidelines**: import NICE / WHO PDFs your institution is licensed to use (Knowledge base tab → Import).
5. **Validate the developer-annotated gold sets** (`data/gold/retrieval_gold_pmc.json`) with a clinician.

## Known limitations / next engineering steps

* CPU real mode uses the 3B VLM and is slow (minutes per study); thesis numbers should come from the 7B model on GPU.
* The report labeller is rule-based (CheXpert-style); a learned labeller (e.g. CheXbert) would give tighter
  clinical-efficacy scores.
* IU X-ray images include some lateral views; the evaluation keeps them (as the providers distribute them).
* Single-node SQLite is fine for research. A hospital deployment would use PostgreSQL and MinIO behind the same
  interfaces.

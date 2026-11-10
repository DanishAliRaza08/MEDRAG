# Evaluation

Run `medrag eval` (or `make eval`, which adds `--strict` so that a failed release gate exits non-zero). Results
are written to `eval_results/results.{json,md}`.

## Metrics

* **Recall@K**: relevant passages in the top K divided by all relevant passages for the query, averaged over
  queries. Relevance is judged at document level, against `data/gold/retrieval_gold.json`.
* **MRR**: mean reciprocal rank of the first relevant passage.
* **Latency**: mean and 95th percentile, per retrieval strategy and per agent variant.
* **Ungrounded-claim (hallucination) rate**: generated non-image claims not supported by a cited retrieved
  passage, divided by all such claims.
* **Report similarity**: semantic similarity to the clinician-style reference report. In real mode this uses a
  biomedical sentence encoder (`MEDRAG_SIM_MODEL`). Offline mode uses a TF-IDF cosine stand-in.
* **Finding accuracy**: whether the expected key finding appears in the generated report.

## Ablations

| Variant | Retrieval | Image reasoning | Self-verification loop |
|---|---|---|---|
| `full` | ✔ | ✔ | ✔ |
| `single_pass` | ✔ | ✔ | ✘ |
| `text_only` | ✔ | ✘ | ✔ |
| `no_retrieval` | ✘ | ✔ | ✘ |

Each variant can also be chosen per analysis in the workspace and the API (`variant`).

## Release gates

These are fixed before the run, in `eval/runner.py:THRESHOLDS`: hybrid+rerank Recall@5 ≥ 0.80, and an
ungrounded-claim rate ≤ 0.10 for the full system. CI runs them on every push.

## Interpreting offline numbers

Offline mode uses a deterministic extractive generator. That generator writes non-image claims by quoting
retrieved evidence, so it is grounded by construction, and `full` and `single_pass` score the same.
The self-correction loop (RQ4) is exercised by tests with deliberately hallucinating generators
(`tests/test_agent.py`). It can only be measured meaningfully with a real generative model (real mode), where
paraphrase and fabrication occur. In the same way, the offline gold set (14 synthetic cases, 26 retrieval queries)
validates the harness and is far too small and easy for research claims.

## For the thesis experiments

1. Replace `data/knowledge_base/` with licensed NICE / WHO / PMC-OA guideline text.
2. Build the retrieval gold set with a clinically knowledgeable annotator: queries paired with the passages
   that should be retrieved.
3. Build the report gold set from held-out MIMIC-CXR / OpenI studies with their reference reports. Never ingest
   those reports into the knowledge base.
4. Run `MEDRAG_MODE=real medrag eval` on a GPU host, and have clinicians review a sample of the drafts.

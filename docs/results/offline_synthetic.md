# MedRAG evaluation (offline mode)

> Offline-mode figures exercise the harness with deterministic stand-ins and synthetic images; they are **not** clinical results.

Corpus: 46 documents, 340 chunks (HashingEmbedder + LexicalReranker).

## Retrieval: sample corpus (26 queries, complete labels)

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Mean ms | P95 ms |
|---|---|---|---|---|---|---|
| dense | 0.904 | 1.000 | 1.000 | 0.974 | 0.07 | 0.10 |
| sparse | 0.942 | 1.000 | 1.000 | 1.000 | 0.01 | 0.03 |
| hybrid | 0.904 | 1.000 | 1.000 | 0.981 | 0.09 | 0.11 |
| hybrid+rerank | 0.942 | 1.000 | 1.000 | 1.000 | 0.93 | 1.00 |

## Retrieval: full corpus incl. PubMed Central (31 developer-annotated queries)

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Mean ms | P95 ms |
|---|---|---|---|---|---|---|
| dense | 0.656 | 0.946 | 0.989 | 0.815 | 0.11 | 0.19 |
| sparse | 0.946 | 1.000 | 1.000 | 0.984 | 0.46 | 0.71 |
| hybrid | 0.946 | 1.000 | 1.000 | 0.984 | 0.65 | 0.81 |
| hybrid+rerank | 0.882 | 1.000 | 1.000 | 0.952 | 5.68 | 6.31 |


Models: embedder `hashing (offline)`, re-ranker `lexical (offline)`, VLM `offline-heuristic-vlm`, generator `offline-extractive`, verifier `Verifier`.

## Clinical drafting (ablations)

| Variant | Ungrounded-claim rate | Report similarity | Finding accuracy | Verified | Iterations | Mean ms |
|---|---|---|---|---|---|---|
| full | 0.000 | 0.531 | 1.000 | 1.000 | 1.07 | 43.1 |
| single_pass | 0.071 | 0.522 | 1.000 | 0.929 | 1.00 | 43.2 |
| text_only | 0.000 | 0.156 | 0.571 | 1.000 | 1.00 | 30.1 |
| no_retrieval | 0.905 | 0.299 | 0.857 | 0.000 | 1.00 | 27.1 |

## Release gates

- retrieval_recall_at_5 (pmc set): **PASS** (value 1.0, threshold 0.8)
- full_hallucination_rate: **PASS** (value 0.0, threshold 0.1)

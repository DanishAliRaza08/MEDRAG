# MedRAG evaluation (offline mode)

> Offline-mode figures exercise the harness with deterministic stand-ins and synthetic images; they are **not** clinical results.

Corpus: 46 documents, 340 chunks (HashingEmbedder + LexicalReranker).

## Retrieval: sample corpus (26 queries, complete labels)

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Mean ms | P95 ms |
|---|---|---|---|---|---|---|
| dense | 0.904 | 1.000 | 1.000 | 0.974 | 0.07 | 0.10 |
| sparse | 0.942 | 1.000 | 1.000 | 1.000 | 0.02 | 0.03 |
| hybrid | 0.904 | 1.000 | 1.000 | 0.981 | 0.10 | 0.13 |
| hybrid+rerank | 0.942 | 1.000 | 1.000 | 1.000 | 1.26 | 1.49 |

## Retrieval: full corpus incl. PubMed Central (31 developer-annotated queries)

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Mean ms | P95 ms |
|---|---|---|---|---|---|---|
| dense | 0.656 | 0.946 | 0.989 | 0.815 | 0.15 | 0.21 |
| sparse | 0.946 | 1.000 | 1.000 | 0.984 | 0.53 | 0.75 |
| hybrid | 0.946 | 1.000 | 1.000 | 0.984 | 0.54 | 0.66 |
| hybrid+rerank | 0.882 | 1.000 | 1.000 | 0.952 | 5.67 | 6.22 |


Models: embedder `hashing (offline)`, re-ranker `lexical (offline)`, VLM `offline-heuristic-vlm`, generator `offline-extractive`, verifier `Verifier`.

## Real chest X-rays: IU X-ray (offline mode, 12 studies with radiologist reports)

| Variant | Precision | Recall | F1 | Normal/abnormal acc. | Ungrounded rate | Similarity | Iterations | Mean s |
|---|---|---|---|---|---|---|---|---|
| full | 0.062 | 0.143 | 0.087 | 0.333 | 0.000 | 0.088 | 1.00 | 0.0 |
| single_pass | 0.062 | 0.143 | 0.087 | 0.333 | 0.000 | 0.088 | 1.00 | 0.0 |
| text_only | 0.167 | 0.286 | 0.210 | 0.250 | 0.000 | 0.034 | 1.00 | 0.0 |
| no_retrieval | 0.062 | 0.143 | 0.087 | 0.333 | 1.000 | 0.077 | 1.00 | 0.0 |

## Release gates

- retrieval_recall_at_5 (pmc set): **PASS** (value 1.0, threshold 0.8)
- full_hallucination_rate: **PASS** (value 0.0, threshold 0.1)

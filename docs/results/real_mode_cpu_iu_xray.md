# MedRAG evaluation (real mode)



Corpus: 46 documents, 340 chunks (BGEM3Embedder + CrossEncoderReranker).

## Retrieval: sample corpus (26 queries, complete labels)

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Mean ms | P95 ms |
|---|---|---|---|---|---|---|
| dense | 0.904 | 1.000 | 1.000 | 0.981 | 104.31 | 132.33 |
| sparse | 0.942 | 1.000 | 1.000 | 1.000 | 0.02 | 0.05 |
| hybrid | 0.942 | 1.000 | 1.000 | 1.000 | 98.48 | 116.44 |
| hybrid+rerank | 0.942 | 1.000 | 1.000 | 1.000 | 4910.71 | 5066.05 |

## Retrieval: full corpus incl. PubMed Central (31 developer-annotated queries)

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Mean ms | P95 ms |
|---|---|---|---|---|---|---|
| dense | 0.946 | 1.000 | 1.000 | 0.984 | 117.04 | 127.49 |
| sparse | 0.946 | 1.000 | 1.000 | 0.984 | 0.44 | 0.69 |
| hybrid | 0.979 | 1.000 | 1.000 | 1.000 | 121.33 | 136.92 |
| hybrid+rerank | 0.946 | 1.000 | 1.000 | 0.984 | 7709.63 | 8124.22 |


Models: embedder `BAAI/bge-m3`, re-ranker `BAAI/bge-reranker-v2-m3`, VLM `qwen2.5-vl`, generator `llm`, verifier `NLIVerifier/v2`.

## Real chest X-rays: IU X-ray (real mode, 12 studies with radiologist reports)

| Variant | Precision | Recall | F1 | Normal/abnormal acc. | Ungrounded rate | Similarity | Iterations | Mean s |
|---|---|---|---|---|---|---|---|---|
| full | 0.083 | 0.143 | 0.105 | 0.250 | 0.278 | 0.928 | 2.75 | 214.7 |
| single_pass | 0.083 | 0.143 | 0.105 | 0.250 | 0.375 | 0.922 | 1.00 | 82.8 |
| text_only | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.934 | 2.00 | 83.9 |
| no_retrieval | 0.083 | 0.143 | 0.105 | 0.250 | 1.000 | 0.915 | 1.00 | 27.8 |

## Release gates

- retrieval_recall_at_5 (pmc set): **PASS** (value 1.0, threshold 0.8)
- full_hallucination_rate: **FAIL** (value 0.2778, threshold 0.1)

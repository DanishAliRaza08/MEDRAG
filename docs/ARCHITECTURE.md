# Architecture

## Layers

| Layer | Module | Responsibility |
|---|---|---|
| Clinical workspace | `medrag/web/` | Dependency-free SPA served by the API at `/`. |
| Application service | `medrag/api/app.py`, `medrag/service.py` | Endpoints for authentication, patients/studies, agent analysis, report lifecycle, search/ask, KB management and audit. `PERMISSIONS` in `api/auth.py` defines role-based access control. |
| Retrieval | `medrag/retrieval/` | Chunking, embedding, dense + sparse search, RRF, re-ranking, metadata pre-filtering, persistence. |
| Agent | `medrag/agent/` | State machine, generator (drafting), verifier (critic). |
| Imaging | `medrag/imaging/` | VLM clients, image encoders, similar-case index, synthetic sample generator. |
| Privacy | `medrag/deid/` | Text PHI scrubbing, image de-identification, pseudonyms, age bands. |
| Reports | `medrag/reports/` | Lifecycle state machine and PDF rendering. |
| Storage | `medrag/storage/` | SQLite records, append-only hash-chained audit log, object store. |
| Evaluation | `medrag/eval/` | Metrics, gold sets, ablation runner, real-X-ray evaluation, release gates, clinician review sheet. |
| Knowledge base | `medrag/kb/` | PubMed Central CC BY corpus builder (`pmc.py`) and PDF/TXT/Markdown guideline importer (`importer.py`). |
| Datasets | `medrag/datasets.py` | IU X-ray real chest radiographs + radiologist reports, rule-based finding labeller with negation. |
| Local model server | `medrag/llm_server.py` | OpenAI-compatible `/v1/chat/completions` for Qwen2.5-VL via `transformers` (real mode without vLLM/GPU). |

## Endpoints

| Method | Path | Permission |
|---|---|---|
| POST | `/api/auth/login` | public |
| GET | `/api/me`, `/api/health` | authenticated / public |
| POST/GET | `/api/patients` | ingest / read |
| POST/GET | `/api/studies`, `/api/studies/{id}`, `/api/studies/{id}/image` | ingest / read |
| POST | `/api/studies/{id}/analyze` (`variant`: full, single_pass, text_only, no_retrieval) | analyze (clinician) |
| GET | `/api/reports`, `/api/reports/{id}`, `/api/reports/{id}/pdf` | read |
| POST | `/api/reports/{id}/signoff`, `/api/reports/{id}/reject` | sign_off (clinician) |
| POST | `/api/search`, `/api/ask` | read |
| GET/POST | `/api/kb`, `/api/kb/documents` | read / kb_write (admin) |
| POST | `/api/kb/upload` (PDF, TXT, Markdown guideline file) | kb_write (admin) |
| GET | `/api/audit`, `/api/audit/verify` | audit (admin) |
| GET | `/api/samples`, `/api/samples/{modality}/{finding}` | read (synthetic demo images) |

Interactive OpenAPI documentation is available at `/docs`.

## Agent control flow

```
route ──► retrieve ──► inspect_image ──► retrieve (query + image concepts) ──► draft ──► verify
              ▲                                                                          │
              └──────────── reformulated query (unsupported concepts) ◄── not passed ────┘
                                                      iterations ≥ max ──► needs_review (flag)
```

* **route** infers the modality, decides whether image inspection is needed, and skips retrieval for the
  no-retrieval baseline.
* **retrieve** runs hybrid search with modality and age-band filters and merges new evidence (deduplicated).
* **inspect_image** calls the VLM and the similar-case index. Image concepts then sharpen the first query.
* **draft** produces claims in four sections. Every non-image claim must cite evidence numbers.
* **verify** checks each claim against its cited passages (support ≥ `MEDRAG_SUPPORT_THRESHOLD`). If any are
  unsupported, it reformulates the query from the missing concepts and loops, up to `MEDRAG_MAX_ITERATIONS`.
  A hard step budget also guards against runaway loops.

The agent's terminal states are `verified`, `needs_review` and `unverified`. `signed_off` is unreachable from
the agent: `reports/lifecycle.transition` rejects any non-human actor, and any human who is not a clinician,
whichever caller invokes it.

`build_langgraph()` compiles the same nodes with LangGraph (`pip install -e ".[graph]"`). The default runner is
dependency-free so that the control flow can be audited and tested anywhere.

## Privacy guarantees (and their tests)

| Guarantee | Mechanism | Test |
|---|---|---|
| No raw patient ID stored | HMAC-SHA256 pseudonym with a secret salt | `test_patient_identifier_stored_only_as_hash` |
| No identifying image metadata | Decode → re-encode from raw pixels | `test_image_metadata_removed` |
| No DICOM header data | Pixels rendered (LUTs, MONOCHROME1); PS3.15 identifying tags + private tags reported and dropped | `test_dicom_header_and_pixels_deidentified` |
| No burned-in text | Local-contrast glyph-stroke detector, text-line growing, optional Tesseract | `test_burned_in_text_removed_without_false_positives` |
| No PHI in notes/questions | Regex scrubber (names, MRN, dates, phone, email, CNIC, SSN, address, age > 89) | `test_scrub_text_categories` |
| No PHI in vector DB | Strict payload allowlist | `test_vector_store_rejects_non_allowlisted_payload` |
| End-to-end | Every persisted byte scanned for the raw identifiers | `test_no_identifiable_data_is_stored` |
| Human sign-off only | Lifecycle state machine below the API | `tests/test_reports.py`, `test_agent_cannot_sign_off_and_attempt_is_audited` |
| Tamper-evident audit | SQL triggers + SHA-256 hash chain | `test_audit_log_is_append_only`, `test_audit_chain_detects_tampering` |

## Verifiers

| Verifier | When | How a claim is judged |
|---|---|---|
| `Verifier` (lexical) | offline mode (default) | cites evidence AND shares ≥ 50% of its key terms with a cited passage |
| `NLIVerifier` | real mode (default), or `MEDRAG_VERIFIER=nli` | an NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`) must find that some 3-sentence window of a cited passage *entails* the claim, and none contradicts it |

The NLI verifier accepts faithful paraphrases that share few words, and it rejects claims that copy words but flip
the meaning (an added or dropped "not"). A contradiction is stored as a negative support score and shown as
"contradicts source" in the workspace and the PDF.

## Running real mode

| Where | Model serving | Command |
|---|---|---|
| GPU server | vLLM (Qwen2.5-VL-7B) + Qdrant | `MEDRAG_EXTRAS=real docker compose --profile gpu up --build` |
| Any machine (CPU) | built-in server (Qwen2.5-VL-3B via `transformers`) | `medrag llm-server --port 8001`, then `MEDRAG_MODE=real MEDRAG_LLM_URL=http://localhost:8001/v1 medrag serve` |

On CPU, the slow steps are capped deliberately: the re-ranker reads at most 512 tokens per passage, embeddings use at
most 1024, the drafting prompt keeps the 4 most relevant sentences of each passage (citations still refer to, and are
verified against, the full passage), and images are resized to at most 256 vision patches.

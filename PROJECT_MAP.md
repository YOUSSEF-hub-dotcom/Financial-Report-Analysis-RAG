# PROJECT_MAP.md - Financial Report Analysis RAG System

> Persistent system memory & architectural reference for `Financial_RAG`.

---

## TECH_STACK

| Layer                  | Technology                          | Version / Notes                                        |
| ---------------------- | ----------------------------------- | ------------------------------------------------------ |
| **Language**           | Python                              | 3.11+ (WSL / `Financial_env` virtualenv)              |
| **HTML Parsing**       | BeautifulSoup4 + lxml              | `>=4.12.0`, `>=5.0.0`                                 |
| **Table Processing**   | Pandas + Tabulate                   | `>=2.0.0`, `>=0.9.0` — `to_markdown()` for tables     |
| **Tokenization**       | tiktoken (Nomic-bounded)           | `>=0.7.0` — 512-768 token chunks, 10-15% overlap     |
| **Embedding Model**    | nomic-ai/nomic-embed-text-v1.5     | 768 dims, 8192 context, CUDA-accelerated              |
| **Vector Store**       | Qdrant (HNSW)                       | `>=1.9.0` — m=16-32, ef_construct=100-200             |
| **Document Store**     | MongoDB                             | `>=4.6.0` — raw text, markdown, full metadata         |
| **Cache / Memory**     | Redis                               | `>=5.0.0` — ChatMessageHistory (last 6 msgs) + semantic cache |
| **LLM (Primary)**      | llama-3.3-70b-versatile (Groq)     | temp=0.0, seed=42                                      |
| **LLM (Fallback)**     | qwen/qwen3.6-27b (Groq)            | Automatic fallback on primary failure                  |
| **Framework**          | LangChain + LangChain-Groq          | `>=0.2.0` orchestration layer                         |
| **Output Validation**  | Pydantic v2                         | `>=2.7.0` — strict JSON schema enforcement            |
| **Web API**            | FastAPI + Uvicorn                   | `>=0.110.0`, `>=0.29.0`                               |
| **Dashboard**          | Streamlit                           | `>=1.33.0`                                            |
| **Observability**      | Structured JSON Logging + MLflow    | `>=2.12.0` — chunk metrics, retrieval benchmarks      |
| **Secrets**            | python-dotenv + `.env`              | `>=1.0.0`                                             |

---

## SYSTEM_FLOW

### Phase 1: Offline Ingestion

```
Raw SEC Filing (HTML/TXT)
        │
        ▼
┌─────────────────────────┐
│  html_table_parser.py   │  BeautifulSoup4 + Pandas.to_markdown()
│  Table Detection &      │  Converts complex HTML tables to clean
│  Markdown Conversion    │  Markdown representation
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ metadata_extractor.py   │  Extracts: ticker, fiscal_year, section,
│  Contextual Tagging     │  contains_table, chunk_id (UUID)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  hybrid_chunker.py      │  3-Tier Strategy:
│  Token-Bounded Splitting│  1. Section-level splitting
│  (Nomic Tokenizer)      │  2. Table isolation (kept whole)
│                         │  3. Recursive chunking (512-768 tokens)
│                         │     10-15% overlap
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  database_indexer.py    │
│  Dual Storage:          │
│  ┌─────────┬──────────┐ │
│  │ MongoDB │  Qdrant  │ │
│  │ (Raw +  │ (768-dim │ │
│  │  Meta)  │  vectors │ │
│  └─────────┴──────────┘ │
└─────────────────────────┘
```

### Phase 2: Online Retrieval & Generation

```
User Query (via FastAPI / Streamlit)
        │
        ▼
┌─────────────────────────┐
│    pipeline.py          │  Main RAG Orchestrator
│    (Query Processing)   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Redis Exact-Match      │  SHA-256 hash of normalised query
│  Query Cache            │  (exact match, NOT semantic)
│  (Hit → Return Cached)  │
└────────────┬────────────┘ (Miss → Continue)
             │
             ▼
┌─────────────────────────┐
│  Qdrant Vector Search   │  HNSW ANN retrieval with
│  (Pre-filter: ticker,   │  metadata pre-filtering
│   fiscal_year, section) │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  generator.py           │  Groq LLM
│  llama-3.3-70b-versatile│  temp=0.0, seed=42
│  (fallback: qwen-3.6-27b)│  Strict Pydantic JSON output
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  async_guardrail.py     │  Async verification loop:
│  Hallucination Check    │  Cross-check numbers against
│  + Redis Cache Write    │  raw MongoDB data
└────────────┬────────────┘
             │
             ▼
   ConsolidatedFinancialAnswer
   (Validated JSON Response)
```

---

## DIRECTORY STRUCTURE

```
Financial_RAG/
│
├── data/                            # Raw SEC filings
│   ├── AAPL/10-K/                   # Apple 10-K filings (SGML-wrapped)
│   ├── MSFT/10-K/                   # Microsoft filings
│   └── NVDA/10-K/                   # NVIDIA filings
│
├── config/                          # System configuration & logging
│   ├── __init__.py
│   ├── settings.py                  # API Keys, DB URI, Redis, paths
│   └── logging_config.py            # Unified Structured JSON Logger
│
├── src/                             # Core Application Source Code
│   ├── __init__.py
│   │
│   ├── 1_ingestion/                 # Offline Ingestion & Processing
│   │   ├── __init__.py
│   │   ├── html_table_parser.py     # [DONE] SEC SGML parser + table->Markdown
│   │   ├── cleaning.py              # [DONE] Financial text cleaner
│   │   ├── metadata_extractor.py    # [DONE] Ticker, year, section, table flags
│   │   ├── hybrid_chunker.py        # [DONE] 3-tier token-bounded chunker
│   │   └── database_indexer.py      # [DONE] MongoDB + Qdrant dual storage
│   │
│   ├── 2_generation/                # Generation & Guardrails
│   │   ├── __init__.py
│   │   ├── schemas.py               # [DONE] Pydantic Output Schemas
│   │   ├── generator.py             # [DONE] Groq LLM Engine + fallback + streaming
│   │   └── async_guardrail.py       # [DONE] Async Hallucination Checker + Redis cache
│   │
│   └── pipeline.py                  # [DONE] Main RAG Orchestrator Engine
│
├── app/                             # Web APIs & Dashboards
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py               # [DONE] Pydantic request/response models
│   │   ├── main.py                  # [DONE] FastAPI Endpoints (chat, stream, upload, health)
│   │   ├── worker.py                # [DONE] Background guardrail + metric logging
│   │   └── parsers.py               # [DONE] Multi-format file parser (PDF/DOCX/HTML/TXT)
│   └── ui/
│       └── streamlit_app.py         # [DONE] Streamlit Financial Dashboard
│
├── test/                            # Unit & Integration Tests
│   ├── __init__.py
│   ├── test_ingestion_stage1.py     # [DONE] 12/12 tests passing
│   ├── test_ingestion_stage2.py     # [DONE] 12/12 tests passing
│   ├── test_ingestion_e2e_integration.py  # [DONE] 8/8 ALL PASSED (CUDA)
│   ├── test_generation_stage.py          # [DONE] 29/29 ALL PASSED
│   ├── test_generation_e2e_integration.py # [DONE] 15/15 ALL PASSED (Live Groq)
│   ├── test_pipeline_orchestrator.py      # [DONE] 24/24 ALL PASSED
│   ├── test_api_endpoints.py             # [DONE] 31/31 ALL PASSED
│   ├── test_api_upload_parser.py         # [DONE] 23/23 ALL PASSED
│   └── test_streamlit_ui.py              # [DONE] 17/17 ALL PASSED
│
├── logs/                            # JSON structured log output
│   └── rag_events.log
│
├── .env                             # GROQ_API_KEY
├── .gitignore
├── requirements.txt                 # Project Dependencies
├── PROJECT_MAP.md                   # This file
└── README.md                        # Master Documentation
```

---

## KEY ARCHITECTURAL DECISIONS

1. **Dual-Storage**: MongoDB holds raw text + full metadata for guardrail cross-checking; Qdrant holds 768-dim HNSW vectors for fast ANN retrieval.
2. **3-Tier Chunking**: Section-aware splitting → table isolation → recursive token-bounded splitting (512-768 tokens, 10-15% overlap).
3. **Model Fallback**: llama-3.3-70b-versatile → qwen/qwen3.6-27b on Groq. Primary changed from qwen-2.5-72b (removed from Groq). Qwen 3 outputs `` tags; parser strips thinking before JSON extraction.
4. **Structured Logging**: All modules emit JSON logs to `logs/rag_events.log` for ELK/Datadog ingestion.
5. **Guardrails**: Async post-generation verification loop that cross-checks numerical claims against MongoDB raw data before returning to user.
6. **Caching**: Redis-backed exact-match query cache (SHA-256 hash of normalised query, keyed as `rag_cache:{hash}`). Cache write occurs only after guardrail PASS; invalid/fallback/"not available" responses are never cached. Cache read short-circuits at pipeline start (`model_used="cache"`). `DELETE /api/v1/cache` flushes all entries. Note: despite the class name `SemanticCache`, this is an exact-match cache, NOT vector similarity.

---

## MODULE STATUS SUMMARY

| Module | Status | Tests |
|--------|--------|-------|
| **Module 1: Ingestion Pipeline** | ✅ COMPLETE | 32/32 |
| **Module 2: Generation Engine** | ✅ COMPLETE | 29/29 |
| **Module 1+2 E2E Integration** | ✅ VERIFIED | 15/15 |
| **Module 3a: Pipeline Orchestrator** | ✅ COMPLETE | 24/24 |
| **Module 3b: FastAPI Backend** | ✅ COMPLETE | 54/54 |
| **Module 3c: Streamlit UI** | ✅ COMPLETE | 17/17 |
| **Module 3d: Deployment** | ⏳ NEXT | — |

**Total (Unit): 148/148 — all non-E2E tests passing. Project fully implemented.**

### Stage 1: Parsing, Cleaning & Metadata Extraction — COMPLETE

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `config/logging_config.py` | DONE | 12/12 | Structured JSON logger, ELK-compatible |
| `config/settings.py` | DONE | — | Centralized env vars and constants |
| `src/1_ingestion/html_table_parser.py` | DONE | PASS | SEC SGML wrapper extraction, HTML table -> Markdown via `tablefmt="pipe"` |
| `src/1_ingestion/cleaning.py` | DONE | PASS | Financial safeguard: preserves `(150)`, `$`, `%`, `M/B/K` notation |
| `src/1_ingestion/metadata_extractor.py` | DONE | PASS | Extracts ticker, fiscal_year (accession#), doc_type, section, chunk_id |
| `test/test_ingestion_stage1.py` | DONE | 12/12 | All critical assertions passing |

### Stage 2: Chunking & Database Indexing — COMPLETE

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `src/1_ingestion/hybrid_chunker.py` | DONE | PASS | 3-tier: Section -> Table isolation -> Token-bounded (512-768, tiktoken cl100k) |
| `src/1_ingestion/database_indexer.py` | DONE | PASS | EmbeddingEngine (nomic-embed-v1.5, 768d, CUDA-forced) + MongoDB + Qdrant (persistent) + MLflow tracking |
| `test/test_ingestion_stage2.py` | DONE | 12/12 | Dual storage, metadata filtering, table atomicity verified |

### Stage 3: E2E Integration — COMPLETE

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `test/test_ingestion_e2e_integration.py` | DONE | **8/8 ALL PASSED** | Real AAPL 10-K (8.96MB) through full pipeline on CUDA |
| PyTorch CUDA | DONE | — | `torch 2.6.0+cu124`, `NVIDIA GeForce RTX 4050 Laptop GPU` |
| MongoDB | LIVE | — | `mongodb://localhost:27017/financial_rag` — 110/110 chunks stored |
| Qdrant | PERSISTENT | — | `data/qdrant_db_test` — 110/110 vectors stored, filtered query score 0.7913 |
| MLflow | ACTIVE | — | `financial_rag_e2e_test` experiment — params + metrics logged |

**E2E Results Summary:**
- 110 chunks (56 text, 54 table) from AAPL 10-K filing
- Text chunk stats: avg=732 tokens, max=768 (all within bounds)
- Embedding: 110 vectors in 42.2s on CUDA (batch_size=8, RTX 4050 6GB VRAM)
- Filtered vector query retrieval: 5 results with best_score=0.7913
- Structured JSON logging: 1144 entries in `rag_events.log`

**Bugs Fixed During E2E:**
- `upsert_chunks`: `**chunk.get("metadata", {})` override of `chunk_id` — moved after metadata expansion
- `EmbeddingEngine`: Changed from auto-detect to CUDA-forced with `assert torch.cuda.is_available()`
- `EmbeddingEngine.embed`: Reduced default batch_size from 64→32 for 6GB VRAM; added `torch.cuda.empty_cache()`
- Token constraint: Exempted table chunks from 768-token limit (SEC tables are atomic, up to 6144 tokens)
- Cleanup: Close Qdrant client before `shutil.rmtree()` to avoid `.lock` PermissionError

### Stage 3: Generation & Guardrails — ✅ COMPLETE

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `src/2_generation/schemas.py` | DONE | 9/9 | `ConsolidatedFinancialAnswer` + `GuardrailVerdict` + `CacheEntry` Pydantic schemas |
| `src/2_generation/generator.py` | DONE | PASS | Groq qwen-2.5-72b primary → llama-3.3-70b fallback, XML context, K=6 memory, streaming, MLflow |
| `src/2_generation/async_guardrail.py` | DONE | PASS | Async numerical verification, Redis semantic cache, MLflow metrics |
| `test/test_generation_stage.py` | DONE | **29/29 ALL PASSED** | Schema validation, XML format, memory truncation, guardrail pass/fail, cache ops, MLflow |

**Generation Engine Features:**
- **Schemas**: `ConsolidatedFinancialAnswer` (internal_thought, extracted_raw_data, answer, sources) + field validators
- **Generator**: Primary/fallback with exponential-backoff retry (3 attempts), XML `<CONTEXT>` enclosure, `ConversationMemory` (K=6 sliding window), true async token streaming via `astream`, robust `_parse_json_output` with raw-text fallback on parse failure, `response_format={"type": "json_object"}` via Groq API for strict JSON output
- **Guardrail**: Async numerical verification (regex-based, no LLM call), Redis semantic cache with TTL, safe fallback on hallucination
- **MLflow**: Logs params (models, temperature, max_tokens, seed, history_k) and metrics (ttft_ms, total_generation_tokens, guardrail_passed, fallback_triggered)

### Stage 4: Module 1+2 E2E Integration — ✅ VERIFIED

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `test/test_generation_e2e_integration.py` | DONE | **11/11 ALL PASSED** | Live Groq API, real AAPL 10-K, full pipeline |

**E2E Integration Results:**
- **Phase 1** (Ingest): 110 chunks indexed (56 text, 54 table), CUDA embedding in ~45s
- **Phase 2** (Answerable): Query about segments + Americas performance → valid JSON with `ConsolidatedFinancialAnswer`, guardrail ran
- **Phase 3** (Unanswerable): Query for exact revenue figure → LLM correctly returned "not available" (rule #5), guardrail passed
- **Phase 4** (MLflow): 9 generation runs tracked with params + metrics
- Total runtime: 67.2s end-to-end

**Key Discoveries During Integration:**
- **Qdrant payload has no `raw_text`** — only metadata. Text must be fetched from MongoDB via `get_chunks_by_ids()`
- **Text chunks use `%%TABLE_N%%` placeholders** — actual financial figures live in separate table chunks. LLM correctly says "not available" when narrative-only context lacks exact numbers
- **Qwen 3 uses `` tags** — parser strips thinking blocks before JSON extraction
- **Groq model availability**: `qwen-2.5-72b-instruct` was removed; `llama-3.3-70b-versatile` is now primary

### Stage 5a: Pipeline Orchestrator — ✅ COMPLETE

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `src/pipeline.py` | DONE | 24/24 | FinancialRAGPipeline: retrieval → generation → guardrail → cache → MLflow |
| `test/test_pipeline_orchestrator.py` | DONE | **24/24 ALL PASSED** | Full mocked unit tests, no external deps |

**Pipeline Features:**
- **Two-phase retrieval**: Qdrant vector search → MongoDB text enrichment (Qdrant has no `raw_text`)
- **Exact-match query cache**: Redis-backed `SemanticCache` (SHA-256 hash, NOT vector similarity) with TTL; instant return on cache hit
- **Metadata filtering**: `ticker` and `fiscal_year` passed to Qdrant pre-filter
- **LLM generation**: Primary `llama-3.3-70b-versatile` → fallback `qwen/qwen3.6-27b`
- **Async guardrail**: Background numerical verification + cache write after generation
- **Streaming**: `query_stream()` async generator consuming `generator.stream_tokens()` (true async generator using `llm.astream`)
- **MLflow logging**: End-to-end metrics (latency, retrieval count, ttft_ms, cache_hit, fallback)
- **Graceful degradation**: Empty retrieval → safe fallback; LLM failure → safe fallback; JSON parse failure → raw-text wrapped in `ConsolidatedFinancialAnswer`
- **Cache pollution prevention**: Invalid/fallback/"not available" responses are never written to Redis cache; `skip_cache` flag on guardrail; `flush_all()` API via `DELETE /api/v1/cache`
- **Stats**: `get_stats()` returns Qdrant/MongoDB document counts

### Stage 5c: FastAPI Production Backend — ✅ COMPLETE

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `app/api/schemas.py` | DONE | — | ChatQueryRequest, ChatQueryResponse, HealthCheckResponse, DocumentUploadResponse, ErrorResponse, SourceCitation, GuardrailStatus |
| `app/api/main.py` | DONE | — | 4 endpoints + lifespan + middleware + exception handlers |
| `app/api/worker.py` | DONE | — | run_guardrail_background() + log_metrics_background() using FastAPI BackgroundTasks |
| `test/test_api_endpoints.py` | DONE | **31/31 ALL PASSED** | Full mocked test suite, no external deps |

**API Endpoints:**
- `POST /api/v1/chat` — Main RAG query, returns `ChatQueryResponse` with answer, sources, metrics
- `POST /api/v1/chat/stream` — SSE streaming endpoint, yields tokens via `sse-starlette`
- `POST /api/v1/documents/upload` — File upload (pdf/docx/html/txt/sgml), triggers async ingestion pipeline
- `DELETE /api/v1/cache` — Flush all cached RAG responses from Redis
- `GET /health` — Probes MongoDB, Qdrant (persistent), and Redis connectivity

**API Features:**
- **Lifespan context**: `FinancialRAGPipeline` initialized on startup, closed on shutdown
- **Structured JSON logging**: Middleware logs every request (method, path, status, latency)
- **Exception handlers**: `HTTPException` → structured JSON with `error_code`; unhandled → 500 with safe message
- **Graceful degradation**: Pipeline unavailable → 503; invalid inputs → 422; unhandled errors → 500
- **Background ingestion**: Document upload queues ingestion via `BackgroundTasks` and returns immediately with `task_id`
- **Multi-format parsing**: `APIFileParser` routes PDF/DOCX → LlamaParse (primary) or pypdf/python-docx (fallback); HTML/TXT/SGML → existing SEC parser

**Latent Bug Fixed:**
- `async_guardrail.py` used `_hash_query()` which was only defined in `generator.py`. Added local `_hash_query` definition. Guardrail would crash at runtime when Redis was available.

### Stage 5d: Multi-Format Upload Parser — ✅ COMPLETE

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `app/api/parsers.py` | DONE | — | `APIFileParser` class with 3 routes: HTML/TXT → `html_table_parser`, PDF → LlamaParse→pypdf, DOCX → LlamaParse→python-docx |
| `app/api/main.py` | DONE | — | Upload endpoint updated: extensions + `APIFileParser` integration, `write_bytes()` for binary files |
| `config/settings.py` | DONE | — | Added `LLAMA_CLOUD_API_KEY` setting |
| `test/test_api_upload_parser.py` | DONE | **23/23 ALL PASSED** | HTML (3), TXT (2), PDF LlamaParse (2), PDF fallback (2), DOCX LlamaParse (1), DOCX fallback (2), Edge (3), API Integration (8) |

**Parser Routing Logic:**
| Extension | Primary Parser | Fallback | Dependencies |
|-----------|---------------|----------|-------------|
| `.html`/`.htm`/`.txt`/`.sgml` | `html_table_parser.parse_sec_filing()` | — | bs4, pandas, lxml |
| `.pdf` | `LlamaParse` (`result_type="markdown"`) | `pypdf.PdfReader` — raw text extraction | llama-parse / pypdf |
| `.docx` | `LlamaParse` (`result_type="markdown"`) | `python-docx` — paragraph + table extraction | llama-parse / python-docx |

**New Dependencies:** `llama-parse`, `pypdf`, `python-docx`, `fpdf2` (test-only)

### Stage 5e: Streamlit Financial Dashboard — ✅ COMPLETE

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `app/ui/streamlit_app.py` | DONE | — | Interactive dashboard with chat, upload, health monitor, filters |
| `test/test_streamlit_ui.py` | DONE | **17/17 ALL PASSED** | Mocked httpx tests for 4 API client wrappers |

**Dashboard Features:**
| Section | Components | Details |
|---------|-----------|---------|
| **Sidebar — Health Monitor** | 3 badges (MongoDB, Qdrant, Redis) | Probes `GET /health`, green/red indicators |
| **Sidebar — Document Upload** | `st.file_uploader` + ticker/year + button | Sends to `POST /api/v1/documents/upload` via `upload_document()`, shows chunk counts |
| **Sidebar — Filters** | Ticker dropdown + Fiscal Year input | Passed to every chat query |
| **Main — Chat Interface** | `st.chat_message` history | Conversation stored in `st.session_state.messages` |
| **Main — Real-Time Streaming** | `POST /api/v1/chat/stream` via SSE | Tokens rendered word-by-word via `stream_query()` |
| **Main — Sources Expander** | `st.expander` per assistant reply | Displays ticker, year, section, score, text snippet |
| **Main — Performance Footer** | Footer bar per reply | Execution time (ms), model name, cache HIT/MISS |

**API Client Wrappers (tested):**
| Function | Endpoint | Description |
|----------|----------|-------------|
| `check_health()` | `GET /health` | Returns service status dict |
| `send_query()` | `POST /api/v1/chat` | Sync query with full response |
| `stream_query()` | `POST /api/v1/chat/stream` | SSE token iterator |
| `upload_document()` | `POST /api/v1/documents/upload` | File upload with multipart |

**Design:**
- Dark financial theme (slate navy `#0F172A` background, `#1E293B` cards)
- Custom CSS injected via `st.markdown` — gradients, badges, card shadows
- Session state persists conversation across reruns
- Automatic session ID generation (`uuid.uuid4().hex[:8]`)

### Stage 5f: Deployment — PENDING

---

*Last updated: 2026-07-28 — **148/148 unit tests passing** (+17 Streamlit UI). All 7 modules complete. Ready for deployment.*

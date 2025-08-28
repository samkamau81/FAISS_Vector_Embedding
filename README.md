# Modular AI Chatbot Platform with GraphQL Retrieval and FAISS Search

A production-grade, plugin-oriented platform for building AI chat experiences that blend **multi-model LLMs** (OpenAI GPT-4, Gemini Pro) with a **GraphQL-driven knowledge retrieval layer**. The system combines **structured queries** with **FAISS-powered semantic search** over unstructured data, and ships with **async I/O**, **Redis caching**, **query batching**, and comprehensive **contract & semantic testing** across Ruby (RSpec) and Python (PyTest).

---

## Key Capabilities

- **Multi-Model LLM Orchestration**  
  Route prompts to GPT-4 or Gemini Pro with policy-based selection, fallbacks, and tool-usage hooks.

- **GraphQL Retrieval Layer**  
  Unified query interface that merges **structured sources** (RDBMS/GraphQL resolvers) with **unstructured corpora** via FAISS vector search.

- **Semantic Search & RAG**  
  FAISS indexes with HNSW/IVF variants, hybrid scoring (BM25 + dense), and chunking strategies for high-recall retrieval augmented generation.

- **Plugin-Oriented Architecture**  
  Independent deployment of conversational capabilities (retrieval, summarization, persona reasoning). Hot-load plugins without touching core runtime.

- **Low-Latency Serving**  
  Async I/O pipelines with **Redis caching** and **query batching**, delivering ~**34% lower average API latency** under **1,000+ concurrent sessions**.

- **Quality Gates & Regression Safety**  
  Cross-language test harness: RSpec for API contract & GraphQL schema checks; PyTest for semantic response validation and regression detection (**250+ tests**).

---

## Architecture Overview
            ┌────────────────────────────────────────────┐
            │              Client Channels               │
            │  Web, Mobile, Slack, Zendesk, Custom UIs  │
            └───────────────────────┬────────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  API Gateway (REST)│
                          │  / GraphQL Router  │
                          └─────────┬──────────┘
                                    │
                ┌───────────────────▼───────────────────┐
                │      Conversation Orchestrator        │
                │  • Policy Routing (GPT-4/Gemini)      │
                │  • Tool/Function Calls                │
                │  • Query Batching & Caching           │
                └───────────┬───────────┬───────────────┘
                            │           │
           ┌────────────────▼─┐     ┌───▼─────────────────┐
           │ Retrieval Plugin │     │ Persona/Reasoning    │
           │ (RAG)           │     │ Plugins               │
           └───────┬─────────┘     └───────────┬──────────┘
                   │                           │
    ┌──────────────▼───────────────┐     ┌─────▼───────────┐
    │ GraphQL Knowledge Layer      │     │ Summarization    │
    │ • Structured resolvers       │     │ • Map/Reduce     │
    │ • Joins & filters            │     │ • Long-context   │
    └──────────────┬───────────────┘     └─────┬───────────┘
                   │                           │
        ┌──────────▼───────────┐     ┌────────▼───────────┐
        │ FAISS Vector Store    │     │ LLM Providers      │
        │ • Dense indices       │     │ • OpenAI GPT-4     │
        │ • Hybrid BM25 + dense │     │ • Gemini Pro       │
        └──────────┬───────────┘     └────────┬───────────┘
                   │                           │
             ┌─────▼───────┐             ┌─────▼─────────┐
             │ Blob Store  │             │ Redis Cache    │
             │ (docs/chunks│             │ + Rate Limits  │
             └─────────────┘             └───────────────┘


---

## Features in Detail

- **Document Processing**
  - Ingestion pipelines (PDF, HTML, Markdown, CSV).
  - Chunking by semantic boundaries; metadata tagging (source, timestamp, access level).
  - Embeddings via OpenAI/Text-Embeddings or local models; FAISS index build & refresh tasks.

- **LLM Applications**
  - Retrieval, summarization (map-reduce & refine), persona-based reasoning.
  - Deterministic tool-calling contracts for GraphQL lookups (schema-validated).
  - Hallucination guards: grounded-only mode, citation injection, and answerability classification.

- **Sentiment & Analytics**
  - Sentiment classification for customer feedback streams.
  - Topic clustering, intent detection, and trend reporting through nightly jobs.

- **Performance**
  - Async HTTP clients; request coalescing; Redis object & embedding cache.
  - Query batching to LLMs and FAISS to reduce cold latency spikes.
  - Horizontal autoscaling of plugin workers.

---

## Tech Stack

- **Languages:** Python, Ruby  
- **APIs/Providers:** OpenAI (GPT-4), Google Gemini Pro  
- **Search/Vector:** FAISS, optional BM25 hybrid  
- **Orchestration:** Async I/O (Python), Sidekiq/Celery (as applicable)  
- **Cache/Queue:** Redis  
- **API:** REST + GraphQL (schema-first, codegen for contracts)  
- **Testing:** RSpec (Ruby), PyTest (Python), golden files for semantic diffs  
- **Ops:** Docker, CI with test matrices, load testing scripts (k6/Locust)

---

## Getting Started

### Prerequisites
- Python 3.10+, Ruby 3.1+
- Docker & Docker Compose
- Redis
- API keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`
 
### Quickstart

```bash
# 1) Clone
git clone https://github.com/your-org/ai-chat-platform.git
cd ai-chat-platform

# 2) Environment
cp .env.example .env
# set OPENAI_API_KEY, GEMINI_API_KEY, REDIS_URL, DB_URL, etc.

# 3) Build & run
docker compose up --build

# 4) Seed FAISS index (example)
make ingest DOCS_PATH=./samples

# 5) Hit the API
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize latest refund policy","persona":"support"}'
```


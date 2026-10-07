# RAG Ticket Search Lab — RAG → Evaluation → Agentic RAG

Portfolio-ready support-ticket AI project built from the supplied Module 4, 5 and 6 exercises.

## Architecture

Embeddings → Chroma Vector Store → Retrieval → LLM Generation → Evaluation → Agentic Tool Selection

## Modules

### 4 — RAG Pipeline
`src/rag_pipeline.py`
- Prompt templates
- Retrieval k
- Metadata filtering
- Ticket citations
- Grounded answers

### 5 — Evaluation
`src/evaluation.py`
- Precision@k
- Recall@k
- F1@k
- Average Precision
- Groundedness judge helper
- Latency measurement

### 6 — Agentic RAG
`src/agentic_rag.py`
- Semantic ticket search
- Exact ticket lookup
- Category search
- Priority search
- Ticket statistics
- LLM tool selection

### Interactive assistant
`python src/chat.py`

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your OpenAI API key.

## Run

```bash
python src/rag_pipeline.py
python src/evaluation.py
python src/agentic_rag.py
python src/chat.py
pytest -q
```

## Evaluation

The evaluation module follows the supplied exercise definitions:
- Precision@k
- Recall@k
- F1@k
- Average Precision
- LLM-as-judge groundedness
- Latency/cost concepts

The supplied material gives example production targets including Precision@3 > 0.80, Recall@3 > 0.70, Groundedness > 0.80 and Completeness > 0.75.

## Agentic RAG

The supplied Agentic RAG material emphasizes that an agent chooses tools based on the query, supports multi-step tasks and can maintain conversational memory. This implementation exposes tools for semantic search, ticket lookup, category, priority and statistics.

## Resume description

**RAG Ticket Search & Agentic Support Assistant**

Built an end-to-end RAG support-ticket system using OpenAI embeddings, LangChain, Chroma and LLM-based generation. Implemented semantic retrieval, metadata filtering, citation-aware prompting, Precision/Recall/F1 evaluation, Average Precision, groundedness evaluation and Agentic RAG with tool selection.

## Source material

The original uploaded exercises are preserved:
- `src/04_rag_pipeline_exercises.md`
- `src/05_evaluation_exercises.md`
- `src/06_agentic_rag_exercises.md`

The included ticket dataset is synthetic and intended for learning/demo use.

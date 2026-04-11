# Self-correcting RAG

A small **retrieval-augmented generation** demo where a **LangGraph** workflow retrieves context, answers a question, **grades** how grounded the answer is in that context, and **rewrites the search query** and retries when the score is too low (up to three retries).

**Stack:** LangGraph, LangChain, Chroma (local vector store), OpenAI (`text-embedding-3-small`, `gpt-3.5-turbo`).

## How it works

1. **retrieve** — Embed `query` and pull the top **k = 3** chunks from Chroma.  
2. **generate** — Answer using only retrieved context and the original **question**.  
3. **grade** — LLM returns a **1–5** grounding score.  
4. **should_retry** — If score **≥ 3** or **retry_count ≥ 3**, stop; else **rewrite** the query and loop back to **retrieve**.

## Prerequisites

- Python 3.10+ recommended  
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```bash
cd self-correcting-rag
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install python-dotenv langchain-openai langchain-chroma \
  langchain-text-splitters langchain-core langgraph chromadb
```

Create a `.env` file in the project root (same folder as the scripts):

```env
OPENAI_API_KEY=sk-...
```

The apps call `load_dotenv()` on that file before creating OpenAI clients. Keep `.env` out of version control.

## Run

From the project directory (so `documents.txt` and `.env` resolve correctly):

```bash
python rag_agent.py
```

**`rag_agent.py`** — Full LangGraph agent: builds the index from `documents.txt`, persists vectors under `./chroma_db`, streams each graph step, then prints the final answer.

**`rag_base.py`** — Simpler linear RAG (load → index → one-shot retrieve + answer). Useful as a baseline without the grade/rewrite loop.

## Project files

| File | Role |
|------|------|
| `documents.txt` | Source text chunked and embedded on startup |
| `chroma_db/` | Local Chroma persistence (created after first run) |
| `.env` | `OPENAI_API_KEY` (do not commit) |

Edit `documents.txt`, delete `chroma_db` if you want a clean re-index, then run again.

## Customize

In `rag_agent.py`, adjust **`initial_state`** (`question` / `query`), chunking (`chunk_size`, `chunk_overlap`), retriever **`k`**, models, or the **score threshold** and **max retries** inside `should_retry`.

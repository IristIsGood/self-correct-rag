
# Self-Correcting RAG: Agentic LangGraph Workflow 🧠
A sophisticated Retrieval-Augmented Generation (RAG) system built with **LangGraph** that implements a self-correction loop. It autonomously evaluates its own performance, grades grounding, and iterates on search queries to eliminate hallucinations.

## 🎯 Problem & Solution
* **Problem:** Standard RAG pipelines are "one-shot" and fragile. If the initial retrieval pulls irrelevant data, the LLM provides a "hallucinated" or ungrounded answer with no way to fix itself.
* **Solution:** A **Reflective Agentic Loop**. By using LangGraph to manage state, the system assesses the quality of its own output. If the "Grounding Score" is too low, the agent autonomously rewrites the search query and re-attempts the retrieval process.

## 🏗️ Technical Architecture & Agentic Logic
This project leverages **LangGraph** to move beyond linear chains and into a cyclic, stateful graph architecture:

1. **The Retrieval Node:** Interacts with a local **ChromaDB** instance to pull the top $k = 3$ document chunks using `text-embedding-3-small`.
2. **The Reflection (Grader) Node:** An LLM-based evaluator that assigns a **1–5 grounding score**. It checks if the generated answer is strictly supported by the retrieved context.
3. **The Rewriter Node (Self-Correction):** If the score fails the threshold, this node analyzes the failure, reformulates the user's question into a more "retrievable" search query, and loops back to the start.
4. **State Management:** The graph maintains a `retry_count` within its state to prevent infinite loops (capped at **3 retries**).

## ✨ Key Features
* **Autonomous Quality Control:** The system refuses to provide low-confidence or ungrounded answers.
* **Query Expansion/Rewriting:** Uses an LLM to "bridge the gap" between user intent and vector database semantics.
* **Local-First Persistence:** Uses **Chroma** for local vector storage, ensuring data stays within the environment.
* **Streaming Graph Steps:** Real-time visibility into the agent's "thought process" as it moves between nodes.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph & LangChain
* **Intelligence:** OpenAI (`gpt-3.5-turbo`, `text-embedding-3-small`)
* **Vector Store:** ChromaDB (Local Persistence)
* **Runtime:** Python 3.10+ 

## 🚀 Getting Started

### 1. Environment Setup
```bash
cd self-correcting-rag
python3 -m venv .venv
source .venv/bin/activate
pip install python-dotenv langchain-openai langchain-chroma \
    langchain-text-splitters langchain-core langgraph chromadb
```

### 2. Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-key-here
```

### 3. Execution
**Run the Agentic Workflow:**
```bash
python rag_agent.py
```
*This script builds the index from `documents.txt`, persists it to `./chroma_db`, and initiates the self-correcting loop.*

## 🔑 Key Technical Decisions
* **Challenge: The "Grounding" Threshold:** Determining when an answer is "good enough" is subjective and prone to error in standard chains. 
* **Technical Fix:** Implemented a prompt-based rubric for the LLM grader that specifically penalizes information *not* found in the retrieved chunks. This ensures the system remains a **closed-domain** solution for maximum accuracy.
* **Graph Topology:** Chose LangGraph over a standard Python loop to allow for **fine-grained state control**, making it easier to track and debug the `retry_count` across different execution branches.

## 🛡️ License
MIT

## 👤 Developer
**Irist** – Building self-healing AI architectures.

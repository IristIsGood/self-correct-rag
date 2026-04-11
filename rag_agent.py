# ============================================================
# SELF-CORRECTING RAG AGENT
# Stack: LangGraph + LangChain + Chroma + OpenAI
#
# Architecture:
#   retrieve → generate → grade → (retry loop) → answer
# ============================================================

from pathlib import Path
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv(Path(__file__).resolve().parent / ".env")

# ── 1. SHARED STATE ─────────────────────────────────────────
# This dict is the "memory" shared across all nodes
class GraphState(TypedDict):
    question: str       # original user question (never modified)
    query: str          # current search query (may be rewritten)
    documents: List[Document]  # retrieved document chunks
    answer: str         # current generated answer
    score: int          # self-evaluation score (1-5)
    retry_count: int    # how many retries so far (max 3)


# ── 2. MODELS & RETRIEVER ───────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Load & chunk documents
with open("documents.txt", "r") as f:
    raw = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = [Document(page_content=c) for c in splitter.split_text(raw)]

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ── 3. PROMPTS ──────────────────────────────────────────────
# Answer prompt: forces the LLM to stay grounded in context
answer_prompt = ChatPromptTemplate.from_template("""
Use ONLY the context below. If the answer isn't there, say so.

Context: {context}
Question: {question}
Answer:""")

# Grade prompt: LLM judges its own answer (1=bad, 5=perfect)
grade_prompt = ChatPromptTemplate.from_template("""
Rate if this answer is grounded in the context (1-5). Return ONLY the number.
1=contradicts context, 3=partially grounded, 5=fully grounded.

Context: {context}
Answer: {answer}
Score:""")

# Rewrite prompt: generates a better search query
rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite this question as a better search query. Return ONLY the rewritten query.
Original: {question}
Better query:""")


# ── 4. NODES ────────────────────────────────────────────────
def retrieve_node(state: GraphState) -> dict:
    # Uses state["query"] (not "question") so rewrites take effect
    docs = retriever.invoke(state["query"])
    return {"documents": docs}

def generate_node(state: GraphState) -> dict:
    context = "\n\n".join([d.page_content for d in state["documents"]])
    msgs = answer_prompt.format_messages(
        context=context, question=state["question"]
    )
    return {"answer": llm.invoke(msgs).content}

def grade_node(state: GraphState) -> dict:
    context = "\n\n".join([d.page_content for d in state["documents"]])
    msgs = grade_prompt.format_messages(
        context=context, answer=state["answer"]
    )
    raw = llm.invoke(msgs).content.strip()
    print(f"\n[GRADE NODE] Raw LLM response: '{raw}'")  # add this line
    try:
        score = int(llm.invoke(msgs).content.strip())
        score = max(1, min(5, score))
    except ValueError:
        score = 1
    print(f"[GRADE NODE] Final score: {score}/5")       # add this line
    return {"score": score}

def rewrite_node(state: GraphState) -> dict:
    # Rewrite query AND increment retry counter
    msgs = rewrite_prompt.format_messages(question=state["question"])
    new_query = llm.invoke(msgs).content.strip()
    return {"query": new_query, "retry_count": state["retry_count"] + 1}


# ── 5. CONDITIONAL EDGE ─────────────────────────────────────
def should_retry(state: GraphState) -> str:
    # Returns the NAME of the next node to route to
    if state["retry_count"] >= 3:   # cap: prevents infinite loops
        return "end"
    elif state["score"] >= 3:       # good enough answer
        return "end"
    else:
        return "rewrite"             # bad answer, try again


# ── 6. BUILD THE GRAPH ──────────────────────────────────────
# This is where we wire everything together
workflow = StateGraph(GraphState)

# Register nodes (name → function)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade",    grade_node)
workflow.add_node("rewrite",  rewrite_node)

# Set entry point (first node to run)
workflow.set_entry_point("retrieve")

# Add linear edges (always go A → B)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade")
workflow.add_edge("rewrite",  "retrieve")  # retry loop!

# Add conditional edge (routing depends on should_retry result)
workflow.add_conditional_edges(
    "grade",        # from this node...
    should_retry,   # ...call this function to decide where to go
    {
        "end":     END,       # "end" → finish the graph
        "rewrite": "rewrite"  # "rewrite" → go to rewrite node
    }
)

# Compile the graph into a runnable app
app = workflow.compile()


# ── 7. RUN IT ───────────────────────────────────────────────
if __name__ == "__main__":

    # BAD question — not in your documents, should score 1-2
    # "question": "What is the capital of France?",
    # "query":    "What is the capital of France?",

    # # GOOD question — directly in your documents, should score 4-5
    # "question": "What is LangGraph?",
    # "query":    "What is LangGraph?",
    initial_state = {
        "question": "What is LangGraph and how does it extend LangChain?",
        "query": "What is LangGraph and how does it extend LangChain?",
        "documents": [],
        "answer": "",
        "score": 0,
        "retry_count": 0
    }

    # stream=True lets us see each node execute in real time
    for step in app.stream(initial_state):
        node_name = list(step.keys())[0]
        print(f"→ Ran node: {node_name}")
        if "score" in step.get(node_name, {}):
            print(f"  Score: {step[node_name]['score']}")
        if "query" in step.get(node_name, {}):
            print(f"  Query: {step[node_name]['query']}")

    final = app.invoke(initial_state)
    print(f"\nFinal answer (score {final['score']}, {final['retry_count']} retries):")
    print(final["answer"])


# RAG
# Retrieval-Augmented Generation. Giving an LLM access to external documents before generating its answer.
# embedding
# A list of ~1,500 numbers representing the semantic meaning of a piece of text. Similar meanings have similar numbers.
# vector store
# A database optimised for storing and searching embeddings. Chroma is a local, file-based vector store.
# StateGraph
# LangGraph's main class. You add nodes (functions) and edges (connections) to it, then compile() it into a runnable app.
# conditional edge
# A routing function in LangGraph that inspects the current state and returns a string: the name of the next node to visit.
# LLM-as-a-judge
# Using an LLM to evaluate the quality of another LLM's output. Cost-effective alternative to human evaluation.
# hallucination
# When an LLM generates confident-sounding but factually wrong information not grounded in its context.
# TypedDict
# Python's way of declaring what keys and value types a dictionary must have. LangGraph uses this for state typing.


# 笔记：Agentic RAG 的核心价值（为什么不直接用 Ctrl+F？）
# 语义挖掘（翻译官）： 突破关键词限制，实现“搜 JVM 调优”能懂“内存管理”的深层语义对齐。

# 逻辑整合（逻辑学家）： 跨文档、跨页码自动推理与总结，直接给答案而非扔一堆 PDF 让用户自己翻。

# 循环自省（质检员）： 通过 LangGraph 闭环实现“自我质检”。发现评分低时，会自动改写问题并重试，而非死板地返回“无结果”。

# 核心总结： 传统搜索是“死索引”，Agentic RAG 是具备自主意识、能自查、会反思的数字专家。
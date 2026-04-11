# ============================================================
# THEORY: Imports
# We import tools from 3 layers:
#   - langchain_openai: connects to OpenAI's API
#   - langchain_chroma: our local vector database
#   - langchain_text_splitters: breaks documents into small chunks
# ============================================================
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(Path(__file__).resolve().parent / ".env")

# ============================================================
# THEORY: Embeddings
# OpenAIEmbeddings converts text → vectors (lists of numbers).
# "text-embedding-3-small" is cheap and accurate.
# The LLM (GPT-3.5) is what generates the actual answer.
# ============================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# temperature=0 means deterministic answers (no randomness)


def load_documents(filepath: str):
    # ============================================================
    # THEORY: Chunking
    # We can't send a 100-page document to GPT-4 — context windows
    # are limited. So we split it into overlapping chunks.
    # chunk_size=500: each chunk is ~500 characters
    # chunk_overlap=50: chunks share 50 chars so context isn't lost
    #                    at chunk boundaries
    # ============================================================
    with open(filepath, "r") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(raw_text)
    return [Document(page_content=chunk) for chunk in chunks]


def build_vectorstore(docs):
    # ============================================================
    # THEORY: Vector store
    # Chroma.from_documents() does two things:
    #   1. Calls OpenAI Embeddings API for each chunk
    #   2. Stores the resulting vectors locally in ./chroma_db/
    # This is the "indexing" phase — done once, reused forever.
    # ============================================================
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore


def build_rag_chain(vectorstore):
    # ============================================================
    # THEORY: Retriever
    # The retriever is a wrapper around the vectorstore.
    # search_kwargs={"k": 3} means: return the 3 most similar chunks.
    # ============================================================
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ============================================================
    # THEORY: Prompt template
    # We give the LLM explicit instructions:
    #   - Use ONLY the context (prevents hallucination)
    #   - If unsure, say so (teaches honesty)
    # The {context} and {question} are filled in at runtime.
    # ============================================================
    prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")

    def rag_chain(question: str) -> str:
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        return response.content, docs  # return docs for grading later

    return rag_chain


# --- Test it ---
if __name__ == "__main__":
    docs = load_documents("documents.txt")
    vectorstore = build_vectorstore(docs)
    chain = build_rag_chain(vectorstore)
    answer, retrieved = chain("What is LangGraph?")
    print(f"Answer: {answer}")
    print(f"\nRetrieved {len(retrieved)} chunks")
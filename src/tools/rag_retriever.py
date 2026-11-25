import os
import shutil
import time

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import EMBEDDING_MODEL

load_dotenv()

DOCS_PATH = "data/curriculum_docs"
CHROMA_PATH = "chroma_db"


def load_docs():
    """Sprint 1 (RAG ingestion): load every .md file as one Document."""
    documents = []
    for file_name in sorted(os.listdir(DOCS_PATH)):
        if file_name.endswith(".md"):
            path = os.path.join(DOCS_PATH, file_name)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            documents.append(Document(page_content=text, metadata={"source": file_name}))
    return documents


def split_docs(documents):
    """Sprint 1 (RAG ingestion): cut documents into 500-token chunks, 50 overlap."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(documents)


def get_embeddings():
    """One place that creates the embedding model, so storing and searching match."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vector_db():
    """Sprint 1 (RAG): load, split, embed and store the chunks in ChromaDB."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)   # start clean so chunks are not stored twice

    documents = load_docs()
    chunks = split_docs(documents)
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )
    print(f"Stored {len(chunks)} chunks in {CHROMA_PATH}")


vector_db = None


def get_vector_db():
    """Open the saved ChromaDB once and reuse it for every search."""
    global vector_db
    if vector_db is None:
        vector_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embeddings(),
        )
    return vector_db


def search_docs(query, k=3):
    """Sprint 1 (RAG): return the k closest chunks and how long the search took."""
    db = get_vector_db()
    start = time.perf_counter()
    results = db.similarity_search(query, k=k)
    milliseconds = (time.perf_counter() - start) * 1000
    return results, milliseconds


def warm_up():
    """Run one throwaway search at startup so the first real question is fast."""
    results, ms = search_docs("warm up")
    return ms


if __name__ == "__main__":
    if not os.path.exists(CHROMA_PATH):
        build_vector_db()

    cold_ms = warm_up()
    print(f"Warm-up (cold start) took {cold_ms:.0f} ms")    

    questions = [
        "What is a router chain?",
        "What is the ReAct loop?",
        "How does chunk overlap work?",
        "Which memory type forgets old messages?",
    ]
    for question in questions:
        results, ms = search_docs(question)
        print(f"{ms:.0f} ms | {question} -> {results[0].metadata['source']}")
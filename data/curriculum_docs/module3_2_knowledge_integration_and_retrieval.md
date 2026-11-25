# Module 3.2: Knowledge Integration and Retrieval (RAG)

Sprints AI and Machine Learning advanced track, Sprint 1 (LangChain).

RAG (Retrieval-Augmented Generation) gives the AI your own documents to look things up in. You store the documents as vectors, retrieve the pieces closest to the question, and put them in the prompt so the answer is grounded in real data.

## Why it matters

- An LLM does not know your private, recent or company-specific data, and it makes things up when unsure.
- RAG fixes this without expensive fine-tuning: the AI reads the relevant text at answer time.
- The result is more accurate, checkable answers, which matters in areas like healthcare.

## The pipeline

RAG has five main stages: ingestion, vectorization, storage, retrieval and generation. On top of that you can add an agent with a fallback tool, and monitoring.

| Stage | What happens |
| --- | --- |
| Ingestion | Load files (CSV, PDF) and split them into chunks |
| Vectorization | An embedding model turns each chunk into a vector |
| Storage | The vectors and their text are saved in a vector database (ChromaDB) |
| Retrieval | The question is embedded and the closest chunks are found (top-k) |
| Generation | The retrieved chunks go into the prompt and the AI writes the answer |

## Key terms

| Term | In plain words |
| --- | --- |
| RAG | Retrieve relevant text from a database first, then let the AI answer using it. |
| Chunk overlap | The part of the text shared between neighbouring chunks, so a sentence or idea cut at a boundary is not lost. |
| Vector embedding | A list of numbers that captures the meaning of a text. Similar meanings have similar vectors. |
| ChromaDB | A free vector database that stores embeddings and finds similar ones. |
| ReAct agent | An agent that reasons and picks tools (see 3.1). Here it can fall back to web search. |
| Langfuse | A free tool that records what your AI app does: speed, tokens, cost and the exact prompts. |

## The six steps

1. Ingestion and splitting: load the files with a document loader, then cut them into chunks with a chosen size and overlap.
2. Embedding: pass each chunk through an embedding model (for example OpenAI Embeddings) to get a vector.
3. Storage: save the vectors and their text in a vector database such as ChromaDB, so you do not need to redo this every time.
4. Retrieval: embed the user's question and find the top-k chunks whose vectors are closest (a similarity measure such as cosine similarity, which compares the direction of two vectors).
5. Grounded generation with a fallback: put the chunks in the prompt for the AI. A ReAct agent can fall back to a web search tool if the database does not have the answer.
6. Observability: attach a tracing tool to watch speed, token use and cost.

## The chunk overlap rule

```
O = S x [0.10, 0.25]
```

Plain words: set the overlap (O) to 10 to 25 percent of the chunk size (S). Example: with chunks of 400 characters, use an overlap between 40 and 100. The demo uses 40, which is 10 percent.

Why overlap at all: if a sentence is cut in half at a chunk boundary, both halves lose meaning. Overlap repeats a little text on both sides of the cut, so context is preserved. Too much overlap just wastes space in the index.

## Demo 1: Loading and splitting

```py
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = CSVLoader(file_path="medical_data.csv")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40
)
chunked_docs = text_splitter.split_documents(documents)
print(f"Created {len(chunked_docs)} chunks from {len(documents)} raw documents.")
```

What each part does:

- `CSVLoader` reads the CSV file and turns each row into a document object.
- `RecursiveCharacterTextSplitter` cuts documents into pieces of at most 400 characters. "Recursive" means it tries to break at natural places first (paragraphs, then lines, then spaces), so it avoids cutting in the middle of words when it can.
- `chunk_overlap=40` repeats 40 characters between neighbouring chunks (10 percent).
- The last line prints how many chunks were created from how many original documents.

## Demo 2: Storing in ChromaDB and searching

```py
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key="your-openai-key")
persist_dir = "chroma_medical_store"

# create the vector database and save it to disk
vector_db = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory=persist_dir
)

# find the 3 chunks closest to the question
query = "I have a severe headache, what are the causes?"
results = vector_db.similarity_search(query, k=3)
for i, doc in enumerate(results, 1):
    print(f"Chunk {i}: {doc.page_content[:120]}...")
```

What each part does:

- `OpenAIEmbeddings` is the model that turns text into vectors.
- `Chroma.from_documents` embeds every chunk and stores the vectors together with the text. `persist_directory` saves everything to a folder on disk, so next time you can load it instead of paying to embed again.
- `similarity_search(query, k=3)` is where text meets vectors. The question is turned into a vector by the same embedding model you stored with, then Chroma returns the 3 closest chunks.
- The loop prints the first 120 characters of each result (`page_content`).
- Search is by meaning, not keywords: the question about "severe headache" can match a chunk about migraines even if the words differ.
- Newer LangChain versions use a separate package for this (`langchain-chroma`).

## Agentic RAG: a fallback for questions the database cannot answer

A plain RAG pipeline always retrieves something, even if nothing relevant exists. An agentic version wraps retrieval as a tool (like in 3.1) and gives the agent a second tool such as web search:

1. The agent first searches the vector database.
2. If the results do not answer the question, it uses the web search tool instead.
3. It then writes the final answer from whichever source worked.

This stops the system from failing, or inventing an answer, when a question falls outside your documents.

## Demo 3: Monitoring with Langfuse

```py
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

langfuse_handler = CallbackHandler(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com"
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
response = llm.invoke(
    [HumanMessage(content="Explain clinical signs of a stroke.")],
    config={"callbacks": [langfuse_handler]}
)
print(response.content)
```

What each part does:

- `CallbackHandler` connects your app to your Langfuse account using your keys.
- Passing it in `config={"callbacks": [...]}` makes every call get recorded. Nothing else in your code changes.
- In the Langfuse dashboard you can then see the speed, the number of tokens, the cost in dollars, and the full prompt that was sent, including hidden prompts that frameworks add for you.
- Why it matters: without it you cannot tell why an answer was slow, wrong or expensive.

## Quick facts

- The AI works with numbers, so text must be turned into embeddings before meaning-based search can happen.
- A 10 to 25 percent overlap keeps context across chunk boundaries without bloating the index.
- Temperature 0 keeps the AI strict and factual.
- Agentic RAG avoids failure by falling back to a tool like Google search when retrieval finds nothing useful.
- Langfuse shows speed, cost and hidden prompts in real time.
- The same embedding model must be used for both storing and searching.

## Common mistakes

| Mistake | Fix |
| --- | --- |
| Splitting documents with no overlap | Use 10 to 25 percent overlap so important context is not cut at chunk boundaries |
| Storing raw text in a normal database and searching by keywords | Use embeddings and a vector database like ChromaDB so search understands meaning |
| A static RAG pipeline that cannot handle questions outside your data | Wrap retrieval in a ReAct agent with a fallback tool such as web search |
| Going to production without tracking prompts and cost | Add Langfuse through callbacks to log prompts, tokens, speed and spend |

## Recap

RAG grounds an AI in your own data: split into overlapping chunks, embed, store in a vector database, retrieve the closest chunks for each question, and answer from them. Add a fallback tool for out-of-scope questions and Langfuse to watch cost and speed.

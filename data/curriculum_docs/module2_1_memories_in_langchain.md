# Module 2.1: Memories in LangChain

Sprints AI and Machine Learning advanced track, Sprint 1 (LangChain).

Memory is the part of a chain that saves the conversation and puts it back into each new prompt, so a stateless LLM can hold a multi-turn conversation.

## Why it matters

- An LLM is stateless. It processes every prompt in isolation, so without memory a chatbot forgets what you said one message ago and gives disconnected or made-up answers.
- Memory fixes this by storing the conversation and injecting it into each new prompt.
- The catch is cost. More history means more tokens, which means more money, slower replies, and eventually a prompt that does not fit in the context window. So the real question is always: what should we keep?

## The four memory types

| Type | What it keeps | Good for | Weak point |
| --- | --- | --- | --- |
| ConversationBufferMemory | The entire chat, unedited | Short chats and quick demos | Cost grows with every turn and it eventually overflows the context window |
| ConversationBufferWindowMemory | Only the last K exchanges | Long chats where recent context matters most | Anything older than K turns is forgotten completely |
| ConversationSummaryMemory | A running summary written by the AI | Long chats where the gist matters | An extra AI call every turn, and exact details (names, numbers) can get lost |
| VectorStoreRetrieverMemory | Every message stored as an embedding, and only the most relevant ones are retrieved | Very long histories with facts scattered around | Needs an embedding model and a vector store, and may miss something if the search misses |

In plain words:

- Buffer is a full recording of the conversation.
- Window is a short-term memory that only holds the last few turns.
- Summary is meeting notes instead of a recording.
- Vector is a searchable archive. You ask "what do we know that relates to this question?" and it pulls out just those pieces.

## Key terms

| Term | In plain words |
| --- | --- |
| ConversationBufferMemory | Stores the full raw history of questions and answers. |
| ConversationBufferWindowMemory | Keeps only the last K exchanges and throws older ones away. |
| ConversationSummaryMemory | Uses an AI call to keep rewriting the history as a short summary. |
| VectorStoreRetrieverMemory | Stores history as vectors and retrieves the top-K pieces closest in meaning to the new question. |
| FAISS | A free library from Facebook for very fast similarity search over vectors. It acts as the vector store here. |
| top-K | A number you choose. In window memory it means how many recent exchanges to keep. In vector memory it means how many similar pieces to retrieve. |

## The window formula

```
ContextSize = Sum(Tokens(Interaction_i)) for i from N-K+1 to N
```

Plain words: with window memory, only the last K exchanges (out of N so far) count toward the size of the history. Example: if each exchange is about 50 tokens and K = 2, the history is about 100 tokens whether the chat is 5 turns or 500 turns long. With buffer memory it would keep growing. That ceiling is the whole point of window memory.

## The four steps

1. Choose a memory type based on how long the chats get and how much you can spend.
2. Create the memory with explicit settings: `memory_key`, `input_key` and `return_messages=True`.
3. Put a placeholder in the prompt that matches the memory (for example `{chat_history}`).
4. Connect the memory, the llm and the prompt in one chain.

What the three settings mean:

- `memory_key`: the name of the variable where the history is inserted into the prompt. The prompt must contain a placeholder with exactly the same name.
- `input_key`: which input holds the user's new message (here `user_input`). The memory needs to know this so it can save the right text.
- `return_messages=True`: the history comes back as message objects (human message, AI message) instead of one long string. Chat-style prompts expect this.

A small illustration of how the names must match:

```py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a travel assistant."),
    MessagesPlaceholder(variable_name="chat_history"),   # same name as memory_key
    ("human", "{user_input}")                            # same name as input_key
])
```

## Demo 1: Window memory (k = 2)

```py
from langchain.memory import ConversationBufferWindowMemory

# keep only the last 2 conversation turns
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    input_key="user_input",
    k=2,
    return_messages=True
)

# pretend a user and the AI have 3 exchanges
memory.save_context({"user_input": "I am going to Paris."}, {"output": "Paris is beautiful!"})
memory.save_context({"user_input": "Should I bring an umbrella?"}, {"output": "Check the forecast, but packing one is safe."})
memory.save_context({"user_input": "What food should I try?"}, {"output": "Try croissants and escargot."})

# look at what the memory still holds (the first exchange is gone)
print(memory.load_memory_variables({})["chat_history"])
```

What happens:

- `save_context(input, output)` stores one exchange: what the user said and what the AI replied.
- Three exchanges were saved but `k=2`, so only the last two remain (umbrella and food). The "I am going to Paris" exchange was dropped.
- `load_memory_variables({})` returns a dictionary. Reading `["chat_history"]` gives the stored messages (2 exchanges = 4 messages).
- The lesson: after this, the AI no longer knows the destination is Paris. That is the weakness of window memory.

## Demo 2: Vector memory with FAISS

```py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory

# put some facts into a vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    ["Client preferences: strict vegan diet", "Client flight leaves Tuesday morning"],
    embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})   # return the top 1 match

# build the memory on top of the retriever
memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="history",
    input_key="user_input"
)

# the memory finds the stored fact that is closest in meaning
context = memory.load_memory_variables({"user_input": "Suggest lunch options for the client."})
print(context["history"])
```

What happens:

- `FAISS.from_texts` turns each text into a vector using the embedding model and stores it.
- `k: 1` means "return only the single closest match".
- When you ask about lunch options, the question is turned into a vector with the same embedding model, and FAISS finds the closest stored vector. That is the vegan diet fact, because lunch, food and diet are related in meaning, even though the words are different.
- The two things to remember: the search is by meaning (not keywords), and the same embedding model must be used for storing and searching.
- Unlike window memory, this can bring back something said 100 turns ago if it is relevant to the new question.
- You need `faiss-cpu` installed and an API key for the embeddings.

## How to choose

| Situation | Good choice |
| --- | --- |
| Short chat or quick demo | Buffer |
| Long chat where mostly recent context matters | Window |
| Long chat where the overall gist matters | Summary |
| Very long history with facts scattered around | Vector |
| A critical fact from the start (like an order number) must survive | Summary, or save the key facts separately in the system prompt. Window would lose it. |

## Quick facts

- Buffer memory's prompt grows with every exchange, so the cost keeps rising.
- Summary memory makes an extra AI call under the hood on every turn to update its summary.
- Vector memory is the basic mechanism behind RAG-style conversational agents.
- `return_messages=True` gives message objects, not one flattened string.

## Common mistakes

| Mistake | Fix |
| --- | --- |
| Using buffer memory for a high-volume customer support bot | Use window or summary memory so you do not exceed context limits and pay more in API costs |
| `memory_key` and the placeholder name in the prompt do not match | Make the name in the prompt braces (like `{chat_history}`) exactly the same as `memory_key` |
| Relying on window memory for a critical fact from the start of the chat | Use summary memory, or keep persistent key facts in the system prompt so they survive past K turns |

## Watch out (outdated code)

These memory classes are deprecated in newer LangChain versions (the warning points to `RunnableWithMessageHistory`). The ideas are the same and still important. The modern approach is memory handled through LangGraph, which is covered in Sprint 2.

## Recap

LLMs forget everything between calls, so memory re-sends the past inside the prompt. Buffer keeps everything, window keeps the last K, summary keeps the gist, and vector keeps whatever is most relevant. Pick based on how long the chats get, what you can afford to forget, and what you can afford to pay.

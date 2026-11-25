# Module 1.1: Foundations of LangChain

Sprints AI and Machine Learning advanced track, Sprint 1 (LangChain).

LangChain is a framework (Python and JavaScript) that gives you ready-made, reusable pieces for building LLM apps: prompt templates, model wrappers, output parsers and memory. You snap them together with the pipe symbol.

## Why it matters

A plain LLM API call has three problems:

- It returns one block of text, not structured data your code can use.
- It remembers nothing between calls (each call starts from zero).
- You rewrite the same glue code (building prompt strings, handling the API payload) in every project.

LangChain solves this the way a Lego set does: each piece does one job, and the pieces snap together. Because every model sits behind the same wrapper, you can switch from OpenAI to Hugging Face or a local model by changing one line.

## Key terms

| Term | In plain words |
| --- | --- |
| PromptTemplate | A prompt with empty slots like `{city}`. You fill the slots later, so you write the prompt once and reuse it for every user. |
| PydanticOutputParser | Tells the AI to reply in JSON matching a schema you defined, then turns the reply into a real Python object. |
| OutputFixingParser | A parser with a backup plan. If the reply does not match the schema, it sends the broken reply back to the AI and asks it to fix it. |
| ConversationChain | A ready-made chatbot chain that has memory built in. |
| ConversationBufferMemory | Memory that saves the full chat transcript and pastes it into each new prompt. |

## The pipe: how the pieces connect

```py
chain = prompt | llm | parser
```

Data flows from left to right: the prompt fills its slots, the llm answers, and the parser turns the answer into an object. This style is called LCEL (LangChain Expression Language). Every piece in the chain is a "runnable", which means it can be run with `.invoke()`.

## The four steps

1. Define the output schema: a Pydantic class listing the fields, their types and a description of each.
2. Set up the parsers: create the `PydanticOutputParser`, then wrap it in `OutputFixingParser.from_llm`.
3. Build the prompt: a template with slots, plus the format instructions filled in using partial variables.
4. Assemble and run: pipe prompt, llm and safe parser together, then call `.invoke()` with the user's values.

## Demo 1: Prompt template and chain

Goal: write the prompt once with slots, then fill the slots with different values.

```py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", api_key="your-api-key")

# a prompt with two empty slots: {destination} and {days}
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("user", "I want to visit {destination} for {days} days. Suggest an itinerary.")
])

chain = prompt | llm   # fill the prompt, then send it to the AI
response = chain.invoke({"destination": "Egypt", "days": 3})
print(response.content)   # .content is just the text of the reply
```

What each part does:

- `from_messages` builds the prompt from a list of messages. `system` sets the AI's role and `user` is the question.
- `{destination}` and `{days}` are slots. The keys you pass to `invoke({...})` must match the slot names exactly.
- The reply comes back as a message object, so you use `.content` to get the text. If you add a `StrOutputParser()` at the end of the chain, you get a plain string instead.
- Do not paste your API key into notebooks you share or push to GitHub. Keep it in a `.env` file instead.

## Demo 2: Structured output with Pydantic and OutputFixingParser

Problem: the AI replies with free text, but your app needs exact fields (a city, a number, some advice). Solution: describe the fields in a schema, tell the AI about it, and check the reply against it.

```py
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

# 1. the "form" the AI must fill in
class WeatherReport(BaseModel):
    city: str = Field(description="city name")
    temperature_c: float = Field(description="temperature in Celsius")
    advice: str = Field(description="what to wear and carry")

llm = ChatOpenAI(model="gpt-4o-mini", api_key="your-api-key")

# 2. a strict parser, plus a safe version that repairs bad replies
base_parser = PydanticOutputParser(pydantic_object=WeatherReport)
safe_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

# 3. the prompt tells the AI the format. partial_variables are filled once, now
prompt = PromptTemplate(
    template="Provide current weather for {city}.\n{format_instructions}",
    input_variables=["city"],
    partial_variables={"format_instructions": safe_parser.get_format_instructions()}
)

# 4. prompt -> AI -> parser
chain = prompt | llm | safe_parser
result = chain.invoke({"city": "London"})
print(f"{result.city}: {result.temperature_c}C - {result.advice}")
```

What each part does:

- `WeatherReport` is the form. `float` means a decimal number. Each `description` is sent to the AI so it knows what goes in that box.
- `get_format_instructions()` writes a paragraph telling the AI: reply in JSON with exactly these fields.
- `input_variables` are the slots filled on every call (`{city}`). `partial_variables` are slots filled once, up front (`{format_instructions}`).
- The parser checks the reply and returns a Python object, so you can write `result.temperature_c` directly.
- The normal parser raises an error if the reply is wrong. The safe parser catches that error, asks the AI to repair its reply, and tries again.

Two layers of protection, used together:

- Format instructions in the prompt prevent the mistake (the AI knows `temperature_c` must be a number).
- OutputFixingParser repairs the mistake after it happens. It costs an extra AI call, so it is a safety net and not a replacement for a good prompt.

## Demo 3: Chatbot with memory

```py
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key="your-api-key")
memory = ConversationBufferMemory()   # saves the full chat

chatbot = ConversationChain(llm=llm, memory=memory, verbose=False)

response1 = chatbot.predict(input="Hi, I want to practice Spanish.")
response2 = chatbot.predict(input="Ask me a question about food.")
print(response2)
```

What each part does:

- The AI remembers nothing by itself. Before each message, the memory pastes the whole chat so far into the prompt, and the AI reads it again from the top.
- `ConversationChain` already has a built-in prompt, so you only give it the llm and the memory.
- `verbose=True` prints the full prompt each turn, which is handy for seeing exactly what the AI receives.
- Memory lives as long as the `memory` object exists. To start fresh, create a new memory object or call `memory.clear()`.
- In a real app, keep one memory object per user, or users will see each other's chats.
- `temperature` controls randomness. Use about 0 for JSON and classification (same answer every time) and about 0.7 for conversation (more natural variety).

## Quick facts

- LangChain works in both Python and JavaScript/TypeScript.
- Model wrappers let you swap OpenAI, Hugging Face or a local model with a one-line change.
- OutputFixingParser makes a second AI call whenever the first reply fails validation.
- Memory options are full buffer, sliding window, summary and vector-backed retrieval.

## Common mistakes

| Mistake | Fix |
| --- | --- |
| Hard-coding input values inside the prompt text | Use a template with `{slots}` so instructions and user input stay separate |
| Using the raw text reply where your code needs structured data | Use `PydanticOutputParser` together with `OutputFixingParser` |
| Using full buffer memory forever in a real chat app | Switch to window or summary memory so the chat does not outgrow the context limit |

## Watch out (outdated code)

`ConversationChain` and `ConversationBufferMemory` are deprecated. The idea (paste the history into the prompt) stays the same, and LangGraph in Sprint 2 is the modern way to do it. Newer LangChain versions may also move some imports.

## Recap

LangChain turns a plain LLM call into reusable parts. Templates give you reusable prompts, Pydantic parsers give you structured output you can trust, and memory gives a chatbot context across turns.

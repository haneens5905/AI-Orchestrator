# Module 1.2: Chaining and Chain Flows in LangChain

Sprints AI and Machine Learning advanced track, Sprint 1 (LangChain).

Chaining means linking prompts, models and parsers into a pipeline. The steps can run one after another (sequential), be sent down one of several paths (router), or run at the same time (parallel).

## Why it matters

One giant prompt that tries to do everything is slow, inconsistent and hard to debug. Splitting the work into small chains has three benefits:

- Each step is simple, so the AI makes fewer mistakes.
- You can send a request to the right specialist chain based on what the user wants.
- Independent tasks can run at the same time, which cuts the waiting time.

## The three flows

| Flow | Idea | Use it when |
| --- | --- | --- |
| Sequential | A, then B. The output of A becomes the input of B. | B needs A's result |
| Router | A classifier picks ONE path, and only that path runs. | Different requests need different specialist chains |
| Parallel | A and B run at the same time on the same input. | The tasks are independent of each other |

Analogies: sequential is an assembly line, router is a hospital reception desk sending you to the right department, parallel is two friends shopping in different parts of the store at once.

## Key terms

| Term | In plain words |
| --- | --- |
| Sequential chain | Steps run in order. Each step passes its validated output to the next. |
| Router chain | First classify the request, then branch to a specialized chain. |
| RunnableParallel | Runs several chains at the same time on the same input and returns all results in one dictionary. |
| RunnableLambda | Wraps an ordinary Python function so it can sit inside a pipe pipeline. Used for small transformations and for routing logic. |
| Retrieval QA chain | A chain used in RAG. It searches a knowledge base and puts the retrieved text into the prompt. |
| PydanticOutputParser | Adds the schema instructions to the prompt and turns the reply into a typed object. |

## The five steps

1. Define Pydantic models and parsers: one "form" per step, so each step has a strict contract.
2. Build prompt templates with slots, and add the format instructions with `.partial`.
3. Build the nodes: each node is prompt, model and parser combined into one chain.
4. Choose how to connect them: pipe them in a line, use `RunnableLambda` for routing, or wrap independent ones in `RunnableParallel`.
5. Run it with `.invoke()` and check the output. Use verbose logging to see what happens at each step.

## Demo 1: Sequential chain (city, then itinerary)

Goal: the user gives a trip goal. Chain 1 picks a city. Chain 2 builds an itinerary for that city.

```py
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

# one "form" per step
class CityResult(BaseModel):
    city: str = Field(description="Selected destination city")

class ItineraryResult(BaseModel):
    activities: list[str] = Field(description="3-day list of itinerary items")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
city_parser = PydanticOutputParser(pydantic_object=CityResult)
itinerary_parser = PydanticOutputParser(pydantic_object=ItineraryResult)

city_prompt = ChatPromptTemplate.from_template(
    "Suggest a city for {goal}.\n{format_instructions}"
).partial(format_instructions=city_parser.get_format_instructions())

itinerary_prompt = ChatPromptTemplate.from_template(
    "Create a 3-day itinerary for {city} focusing on {goal}.\n{format_instructions}"
).partial(format_instructions=itinerary_parser.get_format_instructions())

city_chain = city_prompt | llm | city_parser
itinerary_chain = itinerary_prompt | llm | itinerary_parser

# connect them: build the input that chain 2 needs
sequential_pipeline = (
    {"city": city_chain | (lambda x: x.city), "goal": lambda x: x["goal"]}
    | itinerary_chain
)

sequential_pipeline.invoke({"goal": "adventure hiking"})
```

How the connection works, step by step:

- Input: `{"goal": "adventure hiking"}`.
- The `"city"` key runs chain 1, which returns an object like `CityResult(city="Interlaken")`. The `lambda x: x.city` takes out just the text `"Interlaken"`, because chain 2's `{city}` slot needs plain text, not an object.
- The `"goal"` key passes the original goal along unchanged. Chain 1's output only contains the city, so without this line the goal would be lost and chain 2 would not know the trip is about hiking.
- The dictionary `{"city": "Interlaken", "goal": "adventure hiking"}` fills both slots of chain 2, which returns an `ItineraryResult`.
- A `lambda` is just a tiny one-line function: `lambda x: x.city` means "take x and return x.city".
- `list[str]` in the schema means a list of texts.

## Demo 2: Router (classify, then pick a chain)

Goal: decide what kind of request this is, then send it to only one specialist chain.

```py
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# the classifier may only answer "fun" or "medical"
class RouteQuery(BaseModel):
    travel_type: Literal["fun", "medical"] = Field(description="Travel category")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)   # 0 = same answer every time
router_parser = JsonOutputParser(pydantic_object=RouteQuery)

router_prompt = ChatPromptTemplate.from_template(
    "Classify travel query: {query}\n{format_instructions}"
).partial(format_instructions=router_parser.get_format_instructions())

# two specialist chains
fun_chain = ChatPromptTemplate.from_template("Plan leisure activities for: {query}") | llm
medical_chain = ChatPromptTemplate.from_template("Locate clinics and medical plans for: {query}") | llm

# the reception desk: look at the category, send to the right chain
def route_traffic(info):
    category = info["route"]["travel_type"]
    if category == "fun":
        return fun_chain.invoke({"query": info["query"]})
    return medical_chain.invoke({"query": info["query"]})

classifier_chain = router_prompt | llm | router_parser
full_router = (
    RunnableLambda(lambda x: {"route": classifier_chain.invoke(x), "query": x["query"]})
    | RunnableLambda(route_traffic)
)

full_router.invoke({"query": "I need knee surgery recovery in Germany"})
```

How it works:

- `Literal["fun", "medical"]` limits the classifier to exactly those two answers, so it cannot invent a third category.
- Step 1: the classifier chain returns a dictionary like `{"travel_type": "medical"}`. The first `RunnableLambda` bundles it with the original query as `{"route": ..., "query": ...}`.
- Step 2: `route_traffic` is an ordinary Python `if` statement. It reads the category and runs only the matching chain.
- Use temperature 0 for the classifier so the same query always goes to the same place.

## Demo 3: Parallel (packing list and food guide at once)

Goal: run two independent chains on the same city at the same time.

```py
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

class PackingList(BaseModel):
    items: list[str] = Field(description="List of essential packing items")

class FoodGuide(BaseModel):
    must_try: list[str] = Field(description="Top 5 local foods to try")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
packing_parser = PydanticOutputParser(pydantic_object=PackingList)
food_parser = PydanticOutputParser(pydantic_object=FoodGuide)

packing_chain = (
    ChatPromptTemplate.from_template("Provide packing items for {city}:\n{fmt}")
    .partial(fmt=packing_parser.get_format_instructions())
    | llm | packing_parser
)

food_chain = (
    ChatPromptTemplate.from_template("Recommend 5 dishes in {city}:\n{fmt}")
    .partial(fmt=food_parser.get_format_instructions())
    | llm | food_parser
)

# both chains get the same input and run at the same time
parallel_chain = RunnableParallel(
    packing=packing_chain,
    food=food_chain
)

result = parallel_chain.invoke({"city": "Tokyo"})
# result["packing"] and result["food"] hold the two answers
```

How it works:

- Both chains receive the same input (`{"city": "Tokyo"}`) and run at the same time.
- The result is one dictionary with a key for each branch (`packing` and `food`).
- Speed: if each chain takes 3 seconds, running them one after another takes about 6 seconds, and running them in parallel takes about 3.

## How to choose a flow

- Does step B need step A's output? Use sequential.
- Should the input go to only one of several paths? Use router.
- Are the tasks independent of each other? Use parallel.
- Example: a CV app that writes a summary and also suggests job titles. Neither task needs the other, so use parallel.

## Quick facts

- Temperature 0 on classification (router) nodes keeps the routing consistent.
- The pipe operator `|` is how LCEL combines runnables.
- `RunnableParallel` runs independent AI calls at the same time to reduce total waiting time.
- Pydantic models between steps catch missing or wrong-typed fields before the next step breaks.

## Common mistakes

| Mistake | Fix |
| --- | --- |
| Running independent tasks one after another | Use `RunnableParallel` so they run at the same time |
| High temperature on a router or classifier | Set temperature to 0 so classification is consistent |
| Forgetting the format instructions when using a parser | Add `.partial(format_instructions=parser.get_format_instructions())` to the prompt |
| Passing outputs between chains with hard-coded keys and no validation | Validate in-between results with Pydantic so missing keys are caught early |

## Recap

Chains turn single AI calls into pipelines. Sequential passes results forward, router sends each request to the right specialist, and parallel runs independent work at the same time. Pydantic schemas between the steps keep the handovers clean.

# Module 3.1: Building Your Own Agent

Sprints AI and Machine Learning advanced track, Sprint 1 (LangChain).

An agent is an AI that decides for itself which tools to use and in what order, looping through think, act and observe until it has an answer for the user's goal.

## Why it matters

A chain follows steps you hard-coded. That breaks when a request is unexpected. An agent is given tools and a goal, and it decides what to do at run time. In a chain, you decide the steps. In an agent, the AI decides them.

- Chain: steps are the same for every input. Predictable, cheap and easy to debug.
- Agent: which steps happen, and how many, depends on the input. Flexible, but it can take wrong turns or use more tokens.

## Key terms

| Term | In plain words |
| --- | --- |
| Agent | An AI that decides which tools to call and what to do next based on what it has seen so far. |
| ReAct | Short for Reason + Act. The AI alternates between thinking, using a tool, and reading the result. |
| Agent scratchpad | The agent's working notes inside the prompt. It holds its earlier thoughts, tool calls and results so it knows what it has already done. |
| AgentExecutor | The runner that drives the loop. It calls the AI, runs the tool the AI picked, feeds the result back, and stops when the AI gives a final answer. |
| ConversationBufferMemory | Memory that stores the full chat so the agent remembers earlier turns. |
| Self-Ask with Search | An agent style that breaks a hard question into smaller sub-questions and answers each one with a search tool. |

## The ReAct loop

Each round has three parts:

1. Thought: the AI reasons about what it needs.
2. Action: it picks a tool and the input to give it.
3. Observation: the tool's result is added to the scratchpad.

This repeats until the AI decides it has enough, and then it writes the final answer. An illustrative trace for a London weather question:

```
Thought: I need the current weather in London.
Action: WeatherLookup
Action Input: London
Observation: Rainy, 14C, 82% humidity
Thought: I now know what to recommend.
Final Answer: Wear a waterproof jacket and bring an umbrella.
```

The rule behind it:

```
State_{t+1} = LLM(Query, Memory, Scratchpad_t plus {Thought_t, Action_t, Observation_t})
```

Plain words: at every step, the AI looks at the question, the memory, and everything in the scratchpad so far. Then the newest thought, action and observation are added to the scratchpad, and it goes again. It stops when no more actions are needed.

## The five steps

1. Define the tools. Each tool needs a clear name and description, because the AI reads those to decide when to use it.
2. Set up the AI and the prompt. Use a very low temperature (0). The prompt must include the chat history, the user's question and the agent scratchpad.
3. Build the agent core with a constructor like `create_react_agent`, which combines the AI, the tools and the prompt.
4. Wrap it in an `AgentExecutor` together with the tools and memory. Turn on `verbose=True` to see every step.
5. Run it with the user's question and watch the loop until it returns the final answer.

## Demo 1: Creating a custom tool

A tool is just a normal Python function wrapped with a name and a description.

```py
from langchain.tools import Tool

def calculate_weather_summary(city: str) -> str:
    """Simulates a weather API lookup and returns formatted conditions."""
    mock_data = {"london": "Rainy, 14C, 82% humidity", "cairo": "Sunny, 32C, 25% humidity"}
    return mock_data.get(city.lower(), "Clear, 20C, 50% humidity")

weather_tool = Tool(
    name="WeatherLookup",
    func=calculate_weather_summary,
    description="Useful for fetching the current weather conditions for a given city string."
)
```

What each part does:

- The function takes a city and returns a weather text. Here the data is fake (a small dictionary) to keep the demo simple. In real life it would call a weather API.
- `.get(city.lower(), default)` looks up the city in lowercase and returns the default if the city is not in the dictionary.
- `Tool(...)` wraps the function. `name` is what the AI calls it, `func` is the function that runs, and `description` is how the AI knows when to use it.
- The description is the most important part. A vague description leads to the wrong tool being used. Say what it does, what input it expects, and what it returns.

## Demo 2: Building and running a ReAct agent

```py
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import hub

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [weather_tool]
prompt = hub.pull("hwchase17/react")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

response = executor.invoke({"input": "What should I wear in London today based on the weather?"})
print(response["output"])
```

What each part does:

- `temperature=0` makes the AI pick tools and write inputs the same way every time, which avoids made-up tool names.
- `tools = [weather_tool]` is the list of tools the agent is allowed to use.
- `hub.pull("hwchase17/react")` downloads a ready-made ReAct prompt from the LangChain Hub, so you do not write the Thought/Action/Observation format yourself.
- `create_react_agent` combines the AI, the tools and the prompt into the agent brain.
- `AgentExecutor` runs the loop. `verbose=True` prints each thought, action and observation, which is the best way to see how an agent works.
- `executor.invoke({"input": ...})` starts it. The final answer is in `response["output"]`.

## Multi-agent systems

You can also chain several specialist agents together, each doing one job. Two common layouts:

- Sequential handover: agent A passes its structured output to agent B, then to agent C.
- Hierarchical (manager and workers): a manager agent splits the task and hands pieces to worker agents.

Example: a weekend London trip. One agent searches flights, a second checks the weather, and a third combines both and recommends indoor museum plans.

## Quick facts

- ReAct stands for Reason + Act. It combines reasoning steps with real actions.
- Temperature 0 is standard for agents so that tool choices and arguments are consistent.
- The scratchpad is working memory for a single run: it stores the thoughts, actions and observations so far.
- Multi-agent designs can be sequential handovers or manager and worker layouts.

## Common mistakes

| Mistake | Fix |
| --- | --- |
| Vague tool descriptions | Say exactly when to use the tool, what input it takes, and what it returns |
| High temperature (0.8 or more) for agent planning | Use 0 so the AI does not invent tool names or malformed inputs |
| Leaving `agent_scratchpad` out of the prompt | Always include it, so the AI can see its earlier thoughts and results |

## Watch out (outdated code)

LangChain is moving agents to LangGraph (Sprint 2). `AgentExecutor`-style code is the older approach, and in LangChain 1.x it is replaced by `create_agent`, which runs the same ReAct loop. The idea (think, act, observe, repeat) stays exactly the same.

## Recap

An agent is an AI in a loop. It reasons, picks a tool, reads the result, and repeats until it can answer. Tools with clear descriptions, a low temperature and a scratchpad in the prompt are what make it work. Use a chain when the steps are always the same, and an agent when they depend on the input.

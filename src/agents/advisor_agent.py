from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from src.config import MODEL_NAME
from src.tools.db_queries import sql_tool

load_dotenv()

SYSTEM_PROMPT = (
    "You are the Sprints career advisor. "
    "Use the LearnerDatabase tool to answer questions about learners. "
    "Only use facts that the tool returns. "
    "If the tool returns unable_to_verify, say you cannot verify it. "
    "Never invent learners, scores, salaries or employers."
)


def build_sql_agent():
    """Sprint 1 (ReAct agent): an agent that uses the SQL tool.
    Same loop as the course: Thought, Action, Observation, repeat."""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    tools = [sql_tool]
    agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    return agent


def ask(agent, question):
    """Run the agent and print every step, like verbose=True in the course."""
    result = agent.invoke({"messages": [("user", question)]})
    for message in result["messages"]:
        message.pretty_print()
    return result["messages"][-1].content


if __name__ == "__main__":
    agent = build_sql_agent()
    ask(agent, "What is Mona Ali's target salary?")
    ask(agent, "What is Sara Nabil's GPA?")
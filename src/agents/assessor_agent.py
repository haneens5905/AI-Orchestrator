import json

from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI

from src.config import MODEL_NAME
from src.tools.code_repl import run_python_code

load_dotenv()


class GradeResult(BaseModel):
    """Sprint 1 (Pydantic parsers): the form the Assessor must fill in."""
    score: int = Field(ge=0, le=100, description="grade from 0 to 100")
    feedback_points: list[str] = Field(
        description="short points on what is good or what must be fixed"
    )
    pass_status: Literal["pass", "fail"] = Field(
        description="pass if the score is 60 or more, otherwise fail"
    )


def build_grading_chain():
    """Sprint 1 (prompt templates + output parsers): prompt -> llm -> safe parser."""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    base_parser = PydanticOutputParser(pydantic_object=GradeResult)
    safe_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

    prompt = PromptTemplate(
        template=(
            "You are a strict Python teacher. Grade the student's code.\n"
            "Task: {task}\n"
            "Student code:\n{code}\n"
            "Program output or error:\n{run_output}\n"
            "{format_instructions}"
        ),
        input_variables=["task", "code", "run_output"],
        partial_variables={"format_instructions": base_parser.get_format_instructions()},
    )
    return prompt | llm | safe_parser


def make_run_output(run_result: dict) -> str:
    """Turn the REPL result into plain text for the grading prompt."""
    if run_result["status"] == "blocked":
        return "The code was blocked for safety: " + run_result["reason"]
    if run_result["status"] == "error":
        return "Printed: " + run_result["stdout"] + "\nError: " + run_result["stderr"]
    return "Printed: " + run_result["stdout"]


def grade_submission(task: str, code: str) -> dict:
    """Sprint 1 (REPL tool + Pydantic parser): run the student's code safely,
    then grade it. Returns 'unable_to_verify' if the tools fail."""
    run_result = json.loads(run_python_code(code))

    if run_result["status"] == "unable_to_verify":
        return {"status": "unable_to_verify", "reason": run_result["reason"]}

    run_output = make_run_output(run_result)
    try:
        chain = build_grading_chain()
        grade = chain.invoke({"task": task, "code": code, "run_output": run_output})
    except Exception as e:
        return {"status": "unable_to_verify", "reason": str(e)}

    return {"status": "ok", "grade": grade.model_dump()}


if __name__ == "__main__":
    task = "Write a function add(a, b) that returns the sum, then print add(2, 3)."
    submissions = {
        "correct": "def add(a, b):\n    return a + b\nprint(add(2, 3))",
        "wrong": "def add(a, b):\n    return a - b\nprint(add(2, 3))",
        "crash": "def add(a, b):\n    return a + b\nprint(add(2))",
        "infinite loop": "while True:\n    pass",
    }
    for name, code in submissions.items():
        print(name, "->", grade_submission(task, code))
import json
import os
import subprocess
import sys

from langchain_core.tools import Tool

TIMEOUT_SECONDS = 5
MAX_OUTPUT_CHARS = 2000

# this small program runs inside the separate process.
# it gives the student code only a short list of safe built-in functions.
RUNNER = """
import builtins
import sys

safe_names = [
    "print", "len", "range", "int", "float", "str", "bool", "list", "dict",
    "tuple", "set", "sum", "min", "max", "abs", "round", "sorted", "enumerate",
    "zip", "isinstance", "reversed", "any", "all", "map", "filter",
    "Exception", "ValueError", "TypeError", "ZeroDivisionError", "IndexError",
    "KeyError", "__build_class__",
]
safe_builtins = {}
for name in safe_names:
    safe_builtins[name] = getattr(builtins, name)

code = sys.stdin.read()
exec(code, {"__builtins__": safe_builtins, "__name__": "__main__"})
"""

# words that student code is not allowed to contain
BLOCKED_WORDS = ["__", "import", "open", "exec", "eval", "compile", "globals", "getattr", "input"]


def find_blocked_word(code: str):
    """Return the first blocked word found in the code, or None if the code is clean."""
    for word in BLOCKED_WORDS:
        if word in code:
            return word
    return None


def clean_code(raw_code: str) -> str:
    """Remove markdown code fences, like extract_python_code in Module 2.2."""
    lines = raw_code.strip().split("\n")
    kept = [line for line in lines if not line.startswith("```")]
    return "\n".join(kept).strip()


def run_python_code(code: str) -> str:
    """Sprint 1 (custom tools, Python REPL): run student code safely.
    Returns stdout and errors as JSON, or 'unable_to_verify' if the tool fails."""
    code = clean_code(code)
    blocked = find_blocked_word(code)
    if blocked:
        return json.dumps(
            {"status": "blocked", "reason": f"code contains '{blocked}', which is not allowed"}
        )

    clean_env = {}
    for key in ["SYSTEMROOT", "PATH"]:
        if key in os.environ:
            clean_env[key] = os.environ[key]

    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", RUNNER],
            input=code,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=clean_env,
        )
        if result.returncode == 0:
            return json.dumps(
                {"status": "ok", "stdout": result.stdout[:MAX_OUTPUT_CHARS], "stderr": ""}
            )
        return json.dumps(
            {
                "status": "error",
                "stdout": result.stdout[:MAX_OUTPUT_CHARS],
                "stderr": result.stderr.strip().split("\n")[-1],
            }
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"status": "unable_to_verify", "reason": f"code ran longer than {TIMEOUT_SECONDS} seconds"}
        )
    except Exception as e:
        return json.dumps({"status": "unable_to_verify", "reason": str(e)})


python_repl_tool = Tool(
    name="PythonREPL",
    func=run_python_code,
    description=(
        "Runs a student's Python code in a safe sandbox and returns what it printed "
        "and any error message as JSON. Input: the Python code as text. "
        "Has no file access, no imports, and a 5 second time limit."
    ),
)


if __name__ == "__main__":
    tests = {
        "good code": "def add(a, b):\n    return a + b\nprint(add(2, 3))",
        "with fences": "```python\nprint(sum([3, 7, 10, 15]) / 4)\n```",
        "student bug": "print(1 / 0)",
        "infinite loop": "while True:\n    pass",
        "read .env": 'print(open(".env").read())',
        "import os": "import os\nprint(os.listdir('.'))",
        "class": "class A:\n    x = 5\nprint(A.x)",
        "escape trick": 'for c in ().__class__.__bases__[0].__subclasses__():\n    print(c)',
    }
    for name, code in tests.items():
        print(name, "->", run_python_code(code))
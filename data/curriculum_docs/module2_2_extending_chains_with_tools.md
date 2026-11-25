# Module 2.2: Extending Chains with External Tools

Sprints AI and Machine Learning advanced track, Sprint 1 (LangChain).

Extending a chain with tools means connecting the AI to things it cannot do alone (run code, search the web, query a database, call an API) so its answers are based on real data and not guesses.

## Why it matters

A plain LLM has three limits:

- It only knows what was in its training data, so it has no live information.
- It cannot see your private databases.
- It is unreliable at exact arithmetic and often invents facts when asked for a lookup.

Tools bridge these gaps. The AI becomes the planner and the tool does the exact, verifiable work.

## The core pattern

1. The user asks a question.
2. The AI does not answer directly. It writes an instruction (Python code, a SQL query, a search query, or a URL).
3. Your program cleans that instruction (removes code fences and extra text).
4. The tool runs the instruction and returns a real result.
5. The result goes back to the AI as trusted context, and the AI writes the final answer from it.

The key idea: the AI writes the instruction, your code executes it, and the AI reports what the tool found.

## Tools at a glance

| Tool | What it is | Example use |
| --- | --- | --- |
| Python REPL | An environment that runs Python code the AI wrote and returns the output | Exact calculations, data processing |
| SERP API | A wrapper that searches Google and returns results | Current web information beyond the AI's training cutoff |
| News grounding | Limiting the AI's answer to recent, verified articles found by live search | Questions about recent events |
| RequestsGetTool | Makes an HTTP GET request to a REST API and returns the data | Fetching live structured data (weather, prices) |
| SQL database chain | Turns a natural-language question into a SQL query and runs it on a database | "Who earns the highest salary?" |
| FAISS / VectorStoreQA | Searches a vector database for relevant text and gives it to the AI | RAG over your own documents |

FAISS recap: a library that turns text into vectors and finds the nearest ones quickly, which is what makes meaning-based search possible.

## The five steps

1. Set up the model and the tool. Use a low temperature (0.3 or below), because code and SQL need exact syntax and randomness causes mistakes.
2. Write a prompt that tells the AI to output only the instruction (only the code, only the SQL, only the URL), nothing else.
3. Sanitize the output. Remove code fences, backticks, comments and chatty text so only raw, runnable text is left.
4. Run the tool with the cleaned text (for example `tool.run(...)`).
5. Pass the tool's result into a second chain as context so the AI can write the final, fact-based answer.

## Demo 1: Cleaning and running AI-written Python code

Problem: the AI usually wraps code in markdown fences like a python code block. If you run that text directly, Python gives a syntax error. So you clean it first.

```py
def extract_python_code(raw_output: str) -> str:
    lines = raw_output.strip().split("\n")
    cleaned_lines = [
        line for line in lines
        if not line.startswith("```") and not line.startswith("#")
    ]
    return "\n".join(cleaned_lines).strip()

# pretend this is what the AI wrote
llm_generated_code = """```python
numbers = [3, 7, 10, 15]
average = sum(numbers) / len(numbers)
print(average)
```"""

executable_code = extract_python_code(llm_generated_code)

# run it the way a Python REPL tool does
import io
import sys

output_buffer = io.StringIO()
sys.stdout = output_buffer        # send print() output into a buffer
exec(executable_code, {})         # run the code
sys.stdout = sys.__stdout__       # put print() back to normal

result = output_buffer.getvalue().strip()
print(f"Execution Output: {result}")
```

What happens:

- `extract_python_code` keeps every line except the ones that start with the code fence or with `#` (comments).
- `exec` runs the cleaned code. The code prints the average of 3, 7, 10 and 15, which is 8.75.
- Normally `print` writes to the screen. Here we redirect it into a `StringIO` buffer to capture the output as text, then restore it. The final line prints `Execution Output: 8.75`.
- In the real tool, `PythonREPL.run()` does this work for you.
- Safety note: `exec` runs whatever the AI wrote. In a real app, run AI-generated code in a sandbox and never on a machine that holds secrets.

## Demo 2: Natural language to SQL (SQLite)

```py
import sqlite3

# create a small in-memory database
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()
cursor.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, department TEXT, salary INT)")
cursor.executemany(
    "INSERT INTO employees VALUES (?, ?, ?, ?)",
    [
        (1, "Alice", "Engineering", 90000),
        (2, "Bob", "Marketing", 75000),
        (3, "Charlie", "HR", 60000),
        (4, "Diana", "Engineering", 95000)
    ]
)
conn.commit()

# pretend the AI wrote this SQL for the question: "Who earns the highest salary?"
generated_sql = "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1;"

# clean it, then run it
clean_query = generated_sql.replace("```sql", "").replace("```", "").strip()
cursor.execute(clean_query)
rows = cursor.fetchall()

print(f"Query: {clean_query}")
print(f"Database Result: {rows[0][0]} earns ${rows[0][1]}")
conn.close()
```

What happens:

- The first part builds a tiny database with 4 employees.
- `generated_sql` is simulated here. In a real chain, the AI writes it after seeing the table schema in the prompt along with the user's question.
- `ORDER BY salary DESC LIMIT 1` means: sort from highest salary to lowest and keep only the first row.
- `fetchall()` returns a list of tuples. `rows[0][0]` is the name and `rows[0][1]` is the salary, so the output is: Diana earns $95000.

The real pattern is:

1. Put the table schema and the question in the prompt and ask for SQL only.
2. Clean the SQL.
3. Run it on the database.
4. Give the rows to a second chain that writes the answer in plain English.

The same idea works for the other tools. For web search, the AI writes a search query and the tool returns results. For a REST API, the AI builds the URL and `RequestsGetTool` fetches it. Either way, the result becomes the context for the final answer.

## Real-world example

A university advisory bot combines a SQL tool and a Wikipedia retriever. When a student asks about their GPA and the prerequisites for an advanced physics course, the chain turns the identity lookup into a SQL query against the registrar database, and sends the physics question to the Wikipedia retriever. The final answer combines both verified results.

## Quick facts

- Temperature 0.3 or lower is standard for tool chains, to avoid syntax mistakes in code and SQL.
- A SERP API lets the AI get around its training cutoff by searching the live web.
- Vector stores like FAISS turn text into vectors for nearest-neighbor search by meaning.
- Cleaning functions that strip markdown backticks are required whenever AI output goes straight into an interpreter or database.

## Common mistakes

| Mistake | Fix |
| --- | --- |
| Passing the AI's raw reply straight into an interpreter or database | Strip markdown fences, backticks and conversational text first |
| High temperature (0.8 or more) when generating SQL or API URLs | Use 0.0 to 0.3 for exact, predictable syntax |
| Allowing any GET request without checking the URL or handling network errors | Fix the base URL in the prompt, validate the generated parameters, and catch network exceptions |

## Recap

Tools turn a text predictor into something that can look things up, calculate and query data. The AI writes an instruction, you sanitize it, a tool runs it, and the real result grounds the final answer. Keep the temperature low and never run unsanitized output.

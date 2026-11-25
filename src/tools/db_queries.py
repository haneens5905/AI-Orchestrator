import csv
import sqlite3
import json
from langchain_core.tools import Tool

CSV_PATH = "data/sprints_learner_db.csv"
DB_PATH = "data/sprints_learner_db.db"


def create_db():
    """Sprint 1 (custom tools): turn the learner CSV into a SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS learners")
    cur.execute("""
        CREATE TABLE learners (
            learner_id TEXT,
            name TEXT,
            bootcamp_track TEXT,
            completed_modules INTEGER,
            python_assignment_score INTEGER,
            target_salary_egp INTEGER,
            preferred_employers TEXT,
            readiness_status TEXT
        )
    """)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute(
                "INSERT INTO learners VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["learner_id"],
                    row["name"],
                    row["bootcamp_track"],
                    row["completed_modules"],
                    row["python_assignment_score"],
                    row["target_salary_egp"],
                    row["preferred_employers"],
                    row["readiness_status"],
                ),
            )

    conn.commit()
    conn.close()
    print("Database created")


def run_sql_query(query: str) -> str:
    """Sprint 1 (custom tools): run a read-only SQL query on the learners table.
    Returns the rows as text, or an 'unable to verify' message if anything fails."""
    try:
        query = query.replace("```sql", "").replace("```", "").strip()
        if not query.strip().lower().startswith("select"):
            return json.dumps(
                {"status": "unable_to_verify", "reason": "only SELECT queries are allowed"}
            )

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [col[0] for col in cur.description]
        conn.close()

        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))

        return json.dumps({"status": "ok", "rows": results})

    except Exception as e:
        return json.dumps({"status": "unable_to_verify", "reason": str(e)})


sql_tool = Tool(
    name="LearnerDatabase",
    func=run_sql_query,
    description=(
        "Runs a read-only SQL SELECT query on the Sprints learner database "
        "and returns the rows as JSON. "
        "Table: learners. Columns: learner_id, name, bootcamp_track, "
        "completed_modules, python_assignment_score, target_salary_egp, "
        "preferred_employers, readiness_status. "
        "readiness_status values: 'Ready for Placement', 'In Progress', "
        "'Needs Review', 'Dropped'. "
        "preferred_employers is one text like 'Fawry, Swvl', so search it "
        "with LIKE '%Fawry%'. Input: one SQL SELECT query as text."
    ),
)


if __name__ == "__main__":
    create_db()
    print(run_sql_query("SELECT name, target_salary_egp FROM learners WHERE name = 'Mona Ali'"))
    print(run_sql_query("DELETE FROM learners"))
    print(run_sql_query("SELECT * FROM students"))
    print(sql_tool.invoke("SELECT COUNT(*) AS total FROM learners"))
"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

# `/no_think` disables Qwen3's chain-of-thought so the reply is just the SQL,
# keeping latency down and _extract_sql clean. Drop it if you want the model to
# reason (and pay the token/latency cost).
GENERATE_SQL_SYSTEM = (
    "/no_think\n"
    "You are an expert SQLite analyst. Convert the user's question into a single "
    "SQLite SELECT query.\n"
    "Rules:\n"
    "- Use ONLY the tables and columns in the provided schema. Never invent names.\n"
    "- Output ONLY the SQL inside a ```sql ... ``` fenced block. No prose.\n"
    "- Quote identifiers with double quotes; the schema is quoted the same way.\n"
    "- Prefer the simplest query that answers the question exactly."
)

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = (
    "Schema:\n{schema}\n\n"
    "Question: {question}\n\n"
    "Write the SQLite query."
)


# verify_node passes {question}, {sql}, {result}.
VERIFY_SYSTEM = (
    "/no_think\n"
    "You are a strict SQL reviewer for a text-to-SQL system. Given a question, "
    "the SQL that was run, and its execution result, decide whether the result "
    "plausibly answers the question.\n"
    "Treat as NOT ok: SQL errors, empty results when the question implies rows "
    "exist, obviously wrong columns/aggregation, or a query that ignores part of "
    "the question.\n"
    'Respond with ONLY a JSON object: {{"ok": <true|false>, "issue": "<short '
    'reason, empty if ok>"}}. No prose, no code fences.'
)

VERIFY_USER = (
    "Question: {question}\n\n"
    "SQL:\n{sql}\n\n"
    "Execution result:\n{result}\n\n"
    "Is this result a plausible answer? Reply with the JSON object."
)


# revise_node passes {question}, {schema}, {sql}, {result}, {issue}.
REVISE_SYSTEM = (
    "/no_think\n"
    "You are an expert SQLite analyst fixing a query that failed review. "
    "Given the question, schema, the previous SQL, its execution result, and the "
    "reviewer's complaint, produce a corrected SQLite SELECT query.\n"
    "Rules:\n"
    "- Use ONLY tables/columns in the schema. Quote identifiers with double quotes.\n"
    "- Address the reviewer's complaint specifically.\n"
    "- Output ONLY the SQL inside a ```sql ... ``` fenced block. No prose."
)

REVISE_USER = (
    "Schema:\n{schema}\n\n"
    "Question: {question}\n\n"
    "Previous SQL:\n{sql}\n\n"
    "Execution result:\n{result}\n\n"
    "Reviewer complaint: {issue}\n\n"
    "Write the corrected SQLite query."
)

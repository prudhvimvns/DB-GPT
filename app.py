# app.py
import os
import re
import time
import sqlalchemy as sa
import streamlit as st
from typing import Optional

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from openai import RateLimitError, AuthenticationError

# ----------------------------
# Streamlit page configuration
# ----------------------------
st.set_page_config(page_title="DB GPT", page_icon="🤖")
st.title("DB GPT")

# ----------------------------
# Session state
# ----------------------------
st.session_state.setdefault("history", [])     # list of (user, bot)
st.session_state.setdefault("is_busy", False)  # disable UI while running

# ----------------------------
# Cached resources
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_engine(db_uri: str):
    return sa.create_engine(db_uri)

@st.cache_resource(show_spinner=False)
def get_db(db_uri: str):
    # Minimal schema to keep prompts tiny; used by the agent when needed
    return SQLDatabase.from_uri(
        db_uri,
        sample_rows_in_table_info=0,
        include_tables=None,
    )

def build_agent_lazy(db_uri: str, api_key: str):
    """Build the agent only when we actually need the LLM."""
    os.environ["OPENAI_API_KEY"] = api_key  # ensure LangChain sees it

    db = get_db(db_uri)
    llm = ChatOpenAI(
        model="gpt-4.1-mini",    # fast; you can also try "gpt-4o-mini-high"
        temperature=0,
        max_retries=4,
        timeout=15
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Keep your flags; avoid unsupported ones on older LC if needed
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=6,
        early_stopping_method="generate",
        use_query_checker=False,
        agent_executor_kwargs={"handle_parsing_errors": "Please rephrase your DB question."}
    )

def run_with_backoff(fn, *args, **kwargs):
    delay = 0.8
    for _ in range(4):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            if "insufficient_quota" in str(e).lower():
                raise
            time.sleep(delay)
            delay = min(delay * 2, 6)
    return fn(*args, **kwargs)

# ----------------------------
# SQL post-processing helpers (no pandas)
# ----------------------------
SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

def extract_sql(text: str) -> Optional[str]:
    """If agent printed a SQL SELECT (in a code block or inline), return it."""
    if not text:
        return None
    m = SQL_BLOCK_RE.search(text)
    if m:
        sql = m.group(1).strip()
        return sql if sql.lower().lstrip().startswith("select") else None
    t = text.strip()
    return t if t.lower().startswith("select") else None

def is_safe_select(sql: str) -> bool:
    s = sql.strip().lower()
    banned = ("insert", "update", "delete", "drop", "truncate", "alter", "create", "grant", "revoke")
    return s.startswith("select") and not any(b in s for b in banned)

def run_sql_and_format(engine: sa.Engine, sql: str, user_q: str) -> str:
    """Execute SELECT and return a human-friendly answer (number or rows) without pandas."""
    if not is_safe_select(sql):
        return "Refused to execute non-SELECT SQL."

    with engine.connect() as cx:
        try:
            cx.execute(sa.text("SET LOCAL statement_timeout = 5000"))
        except Exception:
            pass
        result = cx.execute(sa.text(sql))
        rows = result.fetchall()
        cols = list(result.keys())

    # 1x1 -> print number/value
    if len(rows) == 1 and len(cols) == 1:
        val = rows[0][0]
        if "how many" in user_q.lower() or "count" in user_q.lower():
            try:
                return f"**Answer:** {int(val)}"
            except Exception:
                return f"**Answer:** {val}"
        return f"**Answer:** {val}"

    if not rows:
        return "_No rows matched._"

    # Render a small table using Streamlit without pandas
    data = [dict(zip(cols, r)) for r in rows]
    st.dataframe(data, use_container_width=True, hide_index=True)
    return f"Returned **{len(rows)}** row(s)."

# ----------------------------
# Inputs
# ----------------------------
database_uri = st.text_input("Enter your database URI:")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if not (database_uri and openai_api_key):
    st.info("Please enter both database URI and OpenAI API key to start.")
    st.stop()

# Quick DB check (no OpenAI calls)
try:
    engine = get_engine(database_uri)
except Exception as e:
    st.error(f"Database connection error: {e}")
    st.stop()

question = st.text_input("Enter your question:", key="user_question")
ask_clicked = st.button("Ask", disabled=st.session_state["is_busy"])

# ----------------------------
# Fast-path SQL helpers (no LLM)
# ----------------------------
def normalize(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"[?.!]+$", "", q)  # drop trailing punctuation
    q = re.sub(r"\s+", " ", q)
    return q

def fastpath_answer(q: str):
    """Return a quick answer string or None to fall back to LLM."""
    if not q:
        return None
    text = normalize(q)
    engine = get_engine(database_uri)

    # greetings / tiny messages
    if text in {"hi", "hello", "hey", "help"}:
        return "Hi! Ask me about your database: tables, columns, counts, joins, etc."

    # show/list tables
    if text in {"show tables", "list tables"} or "what tables" in text:
        with engine.connect() as cx:
            rows = cx.execute(sa.text("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog','information_schema')
                ORDER BY table_schema, table_name;
            """)).fetchall()
        if not rows:
            return "No user tables found."
        lines = [f"{sch}.{tbl}" for sch, tbl in rows]
        return "Tables:\n- " + "\n- ".join(lines[:200])

    # count tables
    if text in {"how many tables are there", "count tables", "how many tables"}:
        with engine.connect() as cx:
            n = cx.execute(sa.text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog','information_schema');
            """)).scalar()
        return f"There are {n} tables."

    # list columns
    m = re.fullmatch(r"(show|list) columns (in|of) ([\w\.]+)", text)
    if m:
        table = m.group(3)
        if "." in table:
            schema, tbl = table.split(".", 1)
            cond = "table_schema=:schema AND table_name=:tbl"
            params = {"schema": schema, "tbl": tbl}
        else:
            cond = "table_name=:tbl AND table_schema NOT IN ('pg_catalog','information_schema')"
            params = {"tbl": table}
        with engine.connect() as cx:
            cols = cx.execute(sa.text(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE {cond}
                ORDER BY ordinal_position;
            """), params).fetchall()
        if not cols:
            return f"No columns found for '{table}'."
        return "Columns:\n- " + "\n- ".join([f"{c} ({t})" for c, t in cols[:300]])

    # count rows – tolerant patterns like "how many employees are there in employee table"
    m = (
        re.fullmatch(r"(count|how many) rows (in|from) ([\w\.]+)", text)
        or re.fullmatch(r"how many (\w+) (are|are there) (in|on|within) ([\w\.]+) table", text)
        or re.fullmatch(r"how many (records|rows) (are|are there) (in|on|within) ([\w\.]+)", text)
    )
    if m:
        table = m.group(3) if m.re.pattern.startswith("(count") else m.group(4)
        with engine.connect() as cx:
            n = cx.execute(sa.text(f'SELECT COUNT(*) FROM {table}')).scalar()
        return f"**Answer:** {int(n)}"

    return None

# ----------------------------
# Handle submit
# ----------------------------
if ask_clicked and question:
    st.session_state["is_busy"] = True
    try:
        st.session_state["history"].append((question, None))

        # 1) Fast path first (no LLM, no quota needed)
        quick = fastpath_answer(question)
        if quick is not None:
            st.session_state["history"][-1] = (question, quick)
        else:
            # 2) Build agent lazily ONLY when needed (first real NL->SQL query)
            agent_executor = build_agent_lazy(database_uri, openai_api_key)
            with st.spinner("Processing your question..."):
                result = run_with_backoff(agent_executor.invoke, {"input": question})
                answer_text = result.get("output") if isinstance(result, dict) else str(result)

                # If the agent printed SQL, execute it and show RESULT (not SQL)
                sql = extract_sql(answer_text)
                if sql:
                    try:
                        final_answer = run_sql_and_format(get_engine(database_uri), sql, question)
                        st.session_state["history"][-1] = (question, final_answer)
                    except Exception as e:
                        st.session_state["history"][-1] = (question, f"SQL error: {e}")
                else:
                    # otherwise keep the agent's natural-language answer
                    st.session_state["history"][-1] = (question, answer_text)

    except AuthenticationError:
        st.session_state["history"][-1] = (question, "Invalid OpenAI API key.")
    except RateLimitError as e:
        msg = str(e)
        if "insufficient_quota" in msg.lower():
            st.session_state["history"][-1] = (
                question,
                "OpenAI error: **Insufficient quota** for this API key/project. "
                "Add billing/credits or switch to a key with available quota."
            )
        else:
            st.session_state["history"][-1] = (question, f"OpenAI rate limit: {e}")
    except Exception as e:
        st.session_state["history"][-1] = (question, f"Error: {e}")
    finally:
        st.session_state["is_busy"] = False

# ----------------------------
# Chat history (native UI)
# ----------------------------
for u, b in reversed(st.session_state["history"]):
    with st.chat_message("user"):
        st.markdown(u)
    if b is not None:
        with st.chat_message("assistant"):
            st.markdown(b)

left, right = st.columns(2)
with left:
    if st.button("Clear Chat History"):
        st.session_state["history"] = []
        st.rerun()
with right:
    st.caption("Tip: ask natural questions; the app will execute SQL and return the final result.")

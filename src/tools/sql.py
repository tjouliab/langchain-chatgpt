import sqlite3
from pydantic import BaseModel
from typing import List
from langchain.tools import Tool

conn = sqlite3.connect("src/db.sqlite")


def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return ", ".join(row[0] for row in rows if row[0] is not None)


def run_sqlite_query(query: str):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occured: {err}"


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query",
    func=run_sqlite_query,
)


def describe_tables(table_names: str | list):
    c = conn.cursor()
    
    if type(table_names) is list:
        tables = ", ".join(f"'{table}'" for table in table_names)
        rows = c.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables});"
        )
    elif type(table_names) is str:
        rows = c.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_names}';"
        )
    return "\n".join(row[0] for row in rows if row[0] is not None)


describte_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, return the schema of those tables",
    func=describe_tables,
)

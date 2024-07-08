import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
llm_path = os.path.join(base_dir, "..")
sys.path.insert(0, os.path.abspath(llm_path))

from llm import gemini
from tools import sql

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.globals import set_debug, set_verbose

# set_debug(True)
# set_verbose(True)

# Empower Gemini with Tools and Agents
if __name__ == "__main__":
    # Init Model
    model = gemini.model

    # Get the list of the tables formatted
    tables = sql.list_tables()

    # agent_scratchpad is a reserved keyword used by the agent
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content=(
                    "You are an AI that has access to a SQLite database with tables.\n"
                    f"The database has tables of: {tables}\n"
                    "Do not make any assumptions about what tables exist or what columns exist. Instead, use the 'describe_tables' function."
                )
            ),
            HumanMessagePromptTemplate.from_template(template="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Get the list of tools the agent can use
    tools = [sql.run_query_tool, sql.describte_tables_tool]

    # An agent is like a chain that can use tools
    agent = OpenAIFunctionsAgent(llm=model, prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)

    agent_executor("How many users have their shipping address registered ?")
    # agent_executor("What is the name of the user with the longest email adress ?")

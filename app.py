import streamlit as st
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain

import os

st.title("Chat with Database")

database_uri = st.text_input("Enter your database URI:")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

if database_uri and openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
    db = SQLDatabase.from_uri(database_uri)
    llm = OpenAI(temperature=0, verbose=True)

    agent_executor = create_sql_agent(
        llm=OpenAI(temperature=0),
        toolkit=SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0)),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    user_input = st.text_input("Enter your question:")

    if st.button("Ask"):
        if user_input:
            response = agent_executor.run(user_input)
            st.write(f"Chatbot: {response}")  # Display the response string directly
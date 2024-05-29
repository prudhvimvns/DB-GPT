import streamlit as st
from streamlit_chat import message
import psycopg2
import tabulate
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_openai import OpenAI
import os
import pkgutil


st.title("DB GPT")

input_type = 'DB URI'

if input_type == 'DB URI':
    database_uri = st.text_input("Enter your database URI:")
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

    if database_uri and openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        db = SQLDatabase.from_uri(database_uri)
        llm = OpenAI(temperature=0, verbose=True)

        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=db, llm=llm),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        def blank_input():
            st.session_state.user_input = ''

        if 'user_input' not in st.session_state:
            st.session_state.user_input = ''

        # User input textbox
        user_input = st.text_input("Enter your question:", value=st.session_state.user_input)



        if user_input:
            response = agent_executor.run(user_input)
            # Initialize chat history
            if 'history' not in st.session_state:
                st.session_state['history'] = []
            # Append user question and bot response to history
            st.session_state['history'].append((user_input, response))

            # Clear user input
            st.session_state.user_input = ''

        def blank_input():
            st.session_state.user_input = ''

        # If there is chat history, display it after the user input textbox
        if 'history' in st.session_state:
            # Display chat history
            for i in range(len(st.session_state['history'])):
                user_message, bot_response = st.session_state['history'][i]
                message(user_message, is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(bot_response, key=str(i), avatar_style="ai", seed="123")
                st.session_state.user_input = ''
                blank_input()


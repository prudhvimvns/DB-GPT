import os
import streamlit as st
from streamlit_chat import message

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI          # <-- use ChatOpenAI (chat endpoint)
from openai import RateLimitError, AuthenticationError

# Page config
st.set_page_config(page_title="DB GPT", page_icon="🤖")
st.title("DB GPT")

# Session state
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

def blank_input():
    st.session_state.user_input = ''

# UI
database_uri = st.text_input("Enter your database URI:")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

if database_uri and openai_api_key:
    try:
        # Setup OpenAI and Database
        os.environ['OPENAI_API_KEY'] = openai_api_key
        db = SQLDatabase.from_uri(database_uri)

        # ✅ Chat model via chat endpoint (fast option shown)
        llm = ChatOpenAI(model="gpt-4o-mini-high", temperature=0)  # or "gpt-4.1-mini"

        # Create SQL Agent
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=db, llm=llm),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        # User input (no auto-submit on Enter)
        user_input = st.text_input(
            "Enter your question:",
            value=st.session_state.user_input,
            key="input_field",
            on_change=blank_input
        )

        if user_input:
            try:
                with st.spinner('Processing your question...'):
                    # ✅ pass dict to invoke with recent LangChain
                    result = agent_executor.invoke({"input": user_input})
                    # ✅ extract final text
                    bot_text = (
                        result.get("output")
                        if isinstance(result, dict) else str(result)
                    )
                    st.session_state['history'].append((user_input, bot_text))
                    st.session_state.user_input = ''

            except RateLimitError:
                st.error("OpenAI API rate limit exceeded. Please wait a moment before trying again.")
            except AuthenticationError:
                st.error("Invalid OpenAI API key. Please check your API key and try again.")
            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")

        # Display chat history
        if st.session_state['history']:
            for idx, (user_msg, bot_msg) in enumerate(st.session_state['history']):
                message(user_msg, is_user=True, key=f"{idx}_user", avatar_style="big-smile")
                message(bot_msg, key=f"{idx}_bot", avatar_style="ai", seed="123")

    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")

else:
    st.info("Please enter both database URI and OpenAI API key to start.")

# Clear chat
if st.session_state['history']:
    if st.button("Clear Chat History"):
        st.session_state['history'] = []
        st.experimental_rerun()

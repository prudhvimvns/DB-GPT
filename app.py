# import streamlit as st
# from langchain.agents import create_sql_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents import AgentExecutor
# from langchain.agents.agent_types import AgentType
# from langchain.utilities import SQLDatabase
# from langchain.llms import OpenAI
# from langchain_experimental.sql import SQLDatabaseChain

# import os

# st.title("Chat with Database")

# database_uri = st.text_input("Enter your database URI:")
# openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

# if database_uri and openai_api_key:
#     os.environ['OPENAI_API_KEY'] = openai_api_key
#     db = SQLDatabase.from_uri(database_uri)
#     llm = OpenAI(temperature=0, verbose=True)

#     agent_executor = create_sql_agent(
#         llm=OpenAI(temperature=0),
#         toolkit=SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0)),
#         verbose=True,
#         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     )

#     user_input = st.text_input("Enter your question:")

#     if st.button("Ask"):
#         if user_input:
#             response = agent_executor.run(user_input)
#             st.write(f"Chatbot: {response}")  # Display the response string directly





# import pip

# pip.install('SQLDatabase')



# import streamlit as st
# from langchain.agents import create_sql_agent, create_csv_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# from langchain.utilities import SQLDatabase
# from langchain.llms import OpenAI
# from langchain_experimental.sql import SQLDatabaseChain
# import os
# import tempfile

# st.title("DB GPT")

# # Add a radio button for the user to select the input type
# input_type = st.radio('Choose your input type:', ('DB URI', 'CSV'))

# if input_type == 'DB URI':
#     database_uri = st.text_input("Enter your database URI:")
#     openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

#     if database_uri and openai_api_key:
#         os.environ['OPENAI_API_KEY'] = openai_api_key
#         db = SQLDatabase.from_uri(database_uri)
#         llm = OpenAI(temperature=0, verbose=True)

#         agent_executor = create_sql_agent(
#             llm=llm,
#             toolkit=SQLDatabaseToolkit(db=db, llm=llm),
#             verbose=True,
#             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         )

#         user_input = st.text_input("Enter your question:")

#         if st.button("Ask"):
#             if user_input:
#                 response = agent_executor.run(user_input)
#                 st.write(f"Chatbot: {response}")  # Display the response string directly

# elif input_type == 'CSV':
#     uploaded_file = st.file_uploader("Upload a CSV")
#     openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

#     if uploaded_file and openai_api_key:
#         os.environ['OPENAI_API_KEY'] = openai_api_key
#         llm = OpenAI(temperature=0, verbose=True)

#         # Save the uploaded file to a temporary location
#         with tempfile.NamedTemporaryFile(delete=False) as fp:
#             fp.write(uploaded_file.getvalue())
#             temp_path = fp.name

#         # Create the CSV agent using the temporary file path
#         agent = create_csv_agent(
#             llm=llm,
#             path=temp_path,
#             verbose=True,
#             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         )

#         user_input = st.text_input("Enter your question:")

#         if st.button("Ask"):
#             if user_input:
#                 response = agent.run(user_input)
#                 st.write(f"Chatbot: {response}")  # Display the response string directly


import psycopg2
import tabulate
import streamlit as st
from langchain.agents import create_sql_agent,create_csv_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
# from langchain_experimental.sql import SQLDatabaseChain
import os
import tempfile

st.title("DB GPT")

# Add a radio button for the user to select the input type
input_type = st.radio('Choose your input type:', ('DB URI', 'CSV'))

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

        user_input = st.text_input("Enter your question:")

        if st.button("Ask"):
            if user_input:
                response = agent_executor.run(user_input)
                st.write(f"Chatbot: {response}")  # Display the response string directly

elif input_type == 'CSV':
    uploaded_file = st.file_uploader("Upload a CSV")
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

    if uploaded_file and openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        llm = OpenAI(temperature=0, verbose=True)

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(uploaded_file.getvalue())
            temp_path = fp.name

        # Create the CSV agent using the temporary file path
        agent = create_csv_agent(
            llm=llm,
            path=temp_path,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        user_input = st.text_input("Enter your question:")

        if st.button("Ask"):
            if user_input:
                response = agent.run(user_input)
                st.write(f"Chatbot: {response}")  # Display the response string directly





#only DB URI


# import psycopg2
# import tabulate
# import streamlit as st
# # from langchain.agents import create_sql_agent,create_csv_agent
# from langchain.agents import create_sql_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# from langchain.utilities import SQLDatabase
# from langchain.llms import OpenAI
# # from langchain_experimental.sql import SQLDatabaseChain
# import os
# import tempfile

# st.title("DB GPT")
# # Add a radio button for the user to select the input type
# input_type = 'DB URI'

# if input_type == 'DB URI':
#     database_uri = st.text_input("Enter your database URI:")
#     openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

#     if database_uri and openai_api_key:
#         os.environ['OPENAI_API_KEY'] = openai_api_key
#         db = SQLDatabase.from_uri(database_uri)
#         llm = OpenAI(temperature=0, verbose=True)

#         agent_executor = create_sql_agent(
#             llm=llm,
#             toolkit=SQLDatabaseToolkit(db=db, llm=llm),
#             verbose=True,
#             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         )

#         user_input = st.text_input("Enter your question:")

#         if st.button("Ask"):
#             if user_input:
#                 response = agent_executor.run(user_input)
#                 st.write(f"Chatbot: {response}")  # Display the response string directly




# import psycopg2
# import tabulate
# import streamlit as st
# from langchain_experimental.agents import create_sql_agent  # updated import statement
# from langchain_experimental.sql import create_csv_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# from langchain.utilities import SQLDatabase
# from langchain.llms import OpenAI
# import os
# import tempfile

# st.title("DB GPT")

# # Add a radio button for the user to select the input type
# input_type = st.radio('Choose your input type:', ('DB URI', 'CSV'))

# if input_type == 'DB URI':
#     database_uri = st.text_input("Enter your database URI:")
#     openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

#     if database_uri and openai_api_key:
#         os.environ['OPENAI_API_KEY'] = openai_api_key
#         db = SQLDatabase.from_uri(database_uri)
#         llm = OpenAI(temperature=0, verbose=True)

#         agent_executor = create_sql_agent(
#             llm=llm,
#             toolkit=SQLDatabaseToolkit(db=db, llm=llm),
#             verbose=True,
#             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         )

#         user_input = st.text_input("Enter your question:")

#         if st.button("Ask"):
#             if user_input:
#                 response = agent_executor.run(user_input)
#                 st.write(f"Chatbot: {response}")  # Display the response string directly

# elif input_type == 'CSV':
#     uploaded_file = st.file_uploader("Upload a CSV")
#     openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

#     if uploaded_file and openai_api_key:
#         os.environ['OPENAI_API_KEY'] = openai_api_key
#         llm = OpenAI(temperature=0, verbose=True)

#         # Save the uploaded file to a temporary location
#         with tempfile.NamedTemporaryFile(delete=False) as fp:
#             fp.write(uploaded_file.getvalue())
#             temp_path = fp.name

#         # Create the CSV agent using the temporary file path
#         agent = create_csv_agent(
#             llm=llm,
#             path=temp_path,
#             verbose=True,
#             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         )

#         user_input = st.text_input("Enter your question:")

#         if st.button("Ask"):
#             if user_input:
#                 response = agent.run(user_input)
#                 st.write(f"Chatbot: {response}")  # Display the response string directly

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

# ------------------------------------------------


# import psycopg2
# import tabulate
# import streamlit as st
# from langchain.agents import create_sql_agent,create_csv_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# from langchain.utilities import SQLDatabase
# from langchain.llms import OpenAI
# # from langchain_experimental.sql import SQLDatabaseChain
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




# import streamlit as st
# from streamlit_chat import message
# import psycopg2
# import tabulate
# from langchain.agents import create_sql_agent,create_csv_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# from langchain.utilities import SQLDatabase
# from langchain.llms import OpenAI
# import os
# # ... other imports ...

# st.title("DB GPT")

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
#                 # Initialize chat history
#                 if 'history' not in st.session_state:
#                     st.session_state['history'] = []
#                 # Append user question and bot response to history
#                 st.session_state['history'].append((user_input, response))
                
#                 # Display chat history
#                 for i in range(len(st.session_state['history'])):
#                     user_message, bot_response = st.session_state['history'][i]
#                     message(user_message, is_user=True, key=str(i) + '_user', avatar_style="big-smile")
#                     message(bot_response, key=str(i), avatar_style="thumbs")
                
                







# import streamlit as st
# from streamlit_chat import message
# import psycopg2
# import tabulate
# from langchain.agents import create_sql_agent,create_csv_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# from langchain.utilities import SQLDatabase
# from langchain.llms import OpenAI
# import os
# # ... other imports ...

# st.title("DB GPT")

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

#         # If there is chat history, display it before the user input textbox
#         if 'history' in st.session_state:
#             # Display chat history
#             print("RES::🔥🔥🔥🔥🔥🔥🔥",st.session_state)
#             for i in range(len(st.session_state['history'])):
#                 user_message, bot_response = st.session_state['history'][i]
#                 message(user_message, is_user=True, key=str(i) + '_user', avatar_style="big-smile")
#                 message(bot_response, key=str(i), avatar_style="thumbs")

#         if 'user_input' not in st.session_state:
#             st.session_state.user_input = ''

#         # User input textbox
#         user_input = st.text_input("Enter your question:", value=st.session_state.user_input)

#         if st.button("Ask"):
#             if user_input:
#                 response = agent_executor.run(user_input)
#                 # Initialize chat history
#                 print("11111111🔥🔥🔥🔥🔥🔥",response)
#                 print("111111RRRRRRR::👋👋👋👋👋👋",st.session_state)
#                 if 'history' not in st.session_state:
#                     st.session_state['history'] = []
#                 # Append user question and bot response to history
#                 st.session_state['history'].append((user_input, response))
#                 print("2222222RRRRRRR::::",st.session_state)
#                 print("33333333::👋👋👋👋👋👋",st.session_state['history'])
#                 st.session_state.user_input = ''

#                 # Clear user input
#                 st.session_state.user_input = ''
                
#                 # If there is no chat history, display the user input textbox after the chat history
#                 if 'history' not in st.session_state:
#                     user_input = st.text_input("Enter your question:")






# import streamlit as st
# from streamlit_chat import message
# import psycopg2
# import tabulate
# from langchain.agents import create_sql_agent,create_csv_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# from langchain.utilities import SQLDatabase
# from langchain.llms import OpenAI
# import os

# # ... your existing imports ...

# st.title("DB GPT")

# input_type = 'DB URI'

# if input_type == 'DB URI':
#    database_uri = st.text_input("Enter your database URI:")
#    openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password')

#    if database_uri and openai_api_key:
#        os.environ['OPENAI_API_KEY'] = openai_api_key
#        db = SQLDatabase.from_uri(database_uri)
#        llm = OpenAI(temperature=0, verbose=True)

#        agent_executor = create_sql_agent(
#            llm=llm,
#            toolkit=SQLDatabaseToolkit(db=db, llm=llm),
#            verbose=True,
#            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#        )

#        if 'user_input' not in st.session_state:
#            st.session_state.user_input = ''

#        # User input textbox
#        user_input = st.text_input("Enter your question:", value=st.session_state.user_input)

#        if st.button("Ask"):
#            if user_input:
#                response = agent_executor.run(user_input)
#                # Initialize chat history
#                if 'history' not in st.session_state:
#                   st.session_state['history'] = []
#                # Append user question and bot response to history
#                st.session_state['history'].append((user_input, response))

#                # Clear user input
#                st.session_state.user_input = ''

#        # If there is chat history, display it after the user input textbox
#        if 'history' in st.session_state:
#            # Display chat history
#            for i in range(len(st.session_state['history'])):
#                user_message, bot_response = st.session_state['history'][i]
#                message(user_message, is_user=True, key=str(i) + '_user', avatar_style="big-smile")
#                message(bot_response, key=str(i), avatar_style="thumbs")






# import streamlit as st
# import speech_recognition as sr
# from langchain.agents import create_sql_agent, create_csv_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# from langchain.utilities import SQLDatabase
# from langchain.llms import OpenAI
# import os

# # ... your existing imports ...

# st.title("DB GPT")

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

#         # Initialize chat history in session state
#         if 'history' not in st.session_state:
#             st.session_state['history'] = []

#         # Display chat history
#         for message in st.session_state['history']:
#             st.write(f"{message['role']}: {message['content']}")

#         # Button to start recording
#         if st.button("Start Recording"):
#             recognizer = sr.Recognizer()
#             # recording the sound
#             with sr.Microphone() as source:
#                 print("Adjusting noise ")
#                 recognizer.adjust_for_ambient_noise(source, duration=1)
#                 print("Recording for 4 seconds")
#                 recorded_audio = recognizer.listen(source, timeout=4)
#                 print("Done recording")

#             # recognizing the Audio
#             try:
#                 print("Recognizing the text")
#                 user_input = recognizer.recognize_google(
#                     recorded_audio, 
#                     language="en-US"
#                 )
#                 print("Decoded Text : {}".format(user_input))
#             except Exception as ex:
#                 print(ex)

#             if user_input:
#                 # Display user message
#                 st.write(f"user: {user_input}")
#                 # Add user message to chat history
#                 st.session_state['history'].append({"role": "user", "content": user_input})

#                 response = agent_executor.run(user_input)
#                 # Display assistant response
#                 st.write(f"assistant: {response}")
#                 # Add assistant response to chat history
#                 st.session_state['history'].append({"role": "assistant", "content": response})

def build_agent_lazy(db_uri: str, api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key

    db = get_db(db_uri)
    llm = ChatOpenAI(
        model="gpt-4.1-mini",   # or "gpt-4o-mini-high"
        temperature=0,
        max_retries=4,
        timeout=15
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Compatible with older LangChain: no early_stopping_method, no use_query_checker kw
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=6
    )


============


    return create_sql_agent(
        # llm=llm,
        # toolkit=toolkit,
        # verbose=True,
        # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # max_iterations=6
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=6,
        early_stopping_method="generate",
        use_query_checker=False,                      # <- saves one iteration
        agent_executor_kwargs={"handle_parsing_errors": "Please rephrase your DB question."}
    )
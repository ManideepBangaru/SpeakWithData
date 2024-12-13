import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent

def definePandasAgent(llm_model, cached_df, input_query: str):
    pandas_agent = create_pandas_dataframe_agent(
        llm = llm_model, 
        df=cached_df,
        allow_dangerous_code=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False)
    response = pandas_agent.invoke({
        "input": input_query,
        "agent_scratchpad": f"Human: {input_query}\nAI: To answer this question, I need to use Python to analyze the dataframe. I'll use the python_repl_ast tool.\n\nAction: python_repl_ast\nAction Input: ",
        })
    return response
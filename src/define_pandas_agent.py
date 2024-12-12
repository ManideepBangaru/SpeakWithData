import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent

def definePandasAgent(llm_model:object, dataframe):
    pandasAgent = create_pandas_dataframe_agent(
        llm=llm_model,
        df = dataframe,
        allow_dangerous_code=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False
    )
    return pandasAgent
from src.get_csv import getCSV
from src.define_llm import defineLLM
from src.define_pandas_agent import definePandasAgent
from src.load_prompt_template import loadPromptTemplate
from src.define_assistant import defineAssistant
from src.build_langgraph import buildLangGraph
from src.decode_response import decodeResponse
from langchain.tools import Tool
from langgraph.graph.message import MessagesState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Global variable to cache the dataset and its path
cached_df = None
cached_file_path = None

if __name__ == "__main__":
    while True:
        # loading file
        file = input("\nEnter input file path : ")
        query = input("\nEnter your query : ")
        cached_df, cached_file_path = getCSV(file, cached_df, cached_file_path)

        llm = defineLLM("gpt-4o-mini",temperature=0)

        pandasAgent = definePandasAgent(llm, cached_df)
        pandas_tool = Tool(
            name = "PandasAgentTool",
            func = pandasAgent,
            description="Useful for answering questions such as facts, aggregations, trends about a pandas dataframe"
        )

        prompt_template = loadPromptTemplate("data/prompt_template.txt")

        tools = [pandas_tool]
        llm_with_tools = llm.bind_tools(tools)

        # assistant = defineAssistant(llm_with_tools, state)

        sys_msg = SystemMessage(content="You are a helpful assistant tasked with bringing insights and doing mathematical operations on datasets")
        def assistant(state: MessagesState):
            return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

        react_graph = buildLangGraph(tools, assistant)

        messages = [HumanMessage(content=str(prompt_template) + str(query))]
        response = react_graph.invoke({"messages": messages})
        final_response = response["messages"][-1].content
        decoded_response = decodeResponse(final_response)
        print("**********************************************************")
        print(decoded_response)
        print("**********************************************************")
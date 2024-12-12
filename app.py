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
import pandas as pd
import gradio as gr

# Global variable to cache the dataset and its path
cached_df = None
cached_file_path = None

def run_app(file, query):
    global cached_df, cached_file_path
    try:
        # Check if the file path has changed
        if cached_file_path != file.name:
            # Load the uploaded file into a DataFrame
            try:
                cached_df = pd.read_csv(file.name)
                cached_file_path = file.name
                print("dataframe loaded successfully")
            except Exception as e:
                return f"Error loading the dataset: {e}", None
        else:
            print("Using cached dataframe")

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
        return decoded_response["answer"]
    
    except Exception as e:
        return f"Error: {e}", None

# Gradio interface
def gradio_app(file, query):
    if file is None or query.strip() == "":
        return "Please upload a CSV file and provide a query.", None
    response_ = run_app(file, query)
    return response_

# Create Gradio components
with gr.Blocks() as app:
    gr.Markdown("Speak With Data")
    gr.Markdown("Upload a CSV file, enter a query, and get a response based on the data in the file.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"], interactive=True)
            query_input = gr.Textbox(label="Enter Query", placeholder="e.g., What is the average of column X?", lines=2)
            query_button = gr.Button("Submit")
        
        with gr.Column():
            response_output = gr.Markdown(label="Response")

    query_button.click(gradio_app, inputs=[file_input, query_input], outputs=[response_output])

if __name__ == "__main__":
    app.launch()
from src.get_csv import getCSV
from src.define_llm import defineLLM
# from src.define_pandas_agent import definePandasAgent
from src.load_prompt_template import loadPromptTemplate
from src.define_assistant import defineAssistant
from src.build_langgraph import buildLangGraph
from src.decode_response import decodeResponse
# from src.prepare_output import prepareOutput
from src.save_plotly_plot import savePlotlyPlot
from langchain.tools import Tool
from langchain.agents import AgentType
from langgraph.graph.message import MessagesState
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pandas as pd
import gradio as gr
import io
import matplotlib.pyplot as plt
from PIL import Image

# Global variable to cache the dataset and its path
cached_df = None
cached_file_path = None

def run_app(file, query):
    global cached_df, cached_file_path, llm_model
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

        llm_model = defineLLM("gpt-4o-mini",temperature=0)

        # Define the Pandas DataFrame Tool
        def pandasAgent(input_query: str):
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

        pandas_tool = Tool(
            name = "PandasAgentTool",
            func = pandasAgent,
            description="Useful for answering questions such as facts, aggregations, trends about a pandas dataframe"
        )

        prompt_template = loadPromptTemplate("data/prompt_template.txt")

        tools = [pandas_tool]
        llm_with_tools = llm_model.bind_tools(tools)

        sys_msg = SystemMessage(content="You are a helpful assistant tasked with bringing insights")
        def assistant(state: MessagesState):
            return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

        react_graph = buildLangGraph(tools, assistant)

        messages = [HumanMessage(content=str(prompt_template) + str(query))]
        response = react_graph.invoke({"messages": messages})
        final_response = response["messages"][-1].content
        return final_response    
    except Exception as e:
        return f"Error: {e}", None

# Gradio interface
def gradio_app(file, query):
    if file is None or query.strip() == "":
        return "Please upload a CSV file and provide a query.", None
    prepared_response = run_app(file, query)
    # Check for plot generation
    buf = io.BytesIO()
    if plt.get_fignums():  # If Matplotlib figures are generated
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()  # Close to free memory
        # Convert the buffer into a PIL image
        plot_image = Image.open(buf)
        return prepared_response, plot_image
    else:
        return prepared_response, None


# Create Gradio components
with gr.Blocks() as app:
    gr.Markdown("<h1 style='text-align: center'> 🧑‍💻 Speak With Data 👩‍💻 </h1>")
    gr.Markdown("Upload a CSV file, enter a query, and get a response based on the data in the file.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"], interactive=True)
            query_input = gr.Textbox(label="Enter Query", placeholder="e.g., What is the average of column X?", lines=2)
            query_button = gr.Button("Submit")
        
        with gr.Column():
            text_output = gr.Markdown(label="Response (Text)")
            plot_output = gr.Image(label="Generated Plot")
    
    def process_response(file, query):
        response_text, response_plot = gradio_app(file, query)
        # Dynamically update visibility and content
        text_visible = response_text is not None and response_text.strip() != ""
        plot_visible = response_plot is not None
        return (
            gr.update(value=response_text or "No text response generated.", visible=text_visible),
            gr.update(value=response_plot, visible=plot_visible)
        )

    query_button.click(process_response, inputs=[file_input, query_input], outputs=[text_output,plot_output])

if __name__ == "__main__":
    app.launch()
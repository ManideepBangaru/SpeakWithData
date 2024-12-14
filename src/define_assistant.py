from langgraph.graph.message import MessagesState

def defineAssistant(llm_with_tools, state: MessagesState):
    sys_message = "You are a helpful assistant tasked with bringing insights and doing mathematical operations on datasets"
    return {"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}
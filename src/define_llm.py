from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def defineLLM(model_name : str, temperature : float):
    llm_model = ChatOpenAI(
        model = model_name,
        temperature=temperature
    )
    return llm_model
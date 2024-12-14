from langchain_core.prompts import ChatPromptTemplate

def loadPromptTemplate(filepath:str):
    with open(filepath, "r", encoding="utf-8") as file:
        prompt = file.read()
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt
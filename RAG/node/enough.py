from RAG.state import State
from langgraph.graph import END

def check_enough_info(state: State):
    if state["need_old_conversation"] == False and state["need_research"] == False:
        return "LLM_gereration"
    else:
        return END
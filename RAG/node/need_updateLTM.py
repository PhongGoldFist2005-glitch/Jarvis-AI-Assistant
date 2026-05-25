from RAG.state import State
from langgraph.graph import END

def need_update_LTM(state: State):
    if state["relevant_personal_info"]:
        return "Update_LTM"
    else:
        return END
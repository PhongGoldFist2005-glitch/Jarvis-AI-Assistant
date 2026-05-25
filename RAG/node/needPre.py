from RAG.state import State
from langgraph.graph import END

def need_previous(state: State):
    if state["need_old_conversation"]:
        return "Retrieval"
    else:
        return END
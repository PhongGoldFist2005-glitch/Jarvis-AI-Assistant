from RAG.state import State
from langgraph.graph import END

def need_research(state: State):
    if state["need_research"]:
        return "Research"
    else:
        return END
from RAG.state import State

def retrieval_STM(state: State):
    short_term_memory = state["stm"]
    # Lấy tất cả conversation trong STM
    list_conversation = short_term_memory.query_window(query_type= 0)
    return {
        "stm_results": list_conversation
    }
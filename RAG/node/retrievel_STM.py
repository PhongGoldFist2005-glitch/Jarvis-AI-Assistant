from RAG.state import State

def retrieval_STM(state: State):
    short_term_memory = state["stm"]
    # Lấy tất cả conversation trong STM
    list_conversation = short_term_memory.query_window(query_type= 0)
    print("[retrieval_STM] count:", 0 if list_conversation is None else len(list_conversation))
    if list_conversation:
        sample = list_conversation[:3]
        print("[retrieval_STM] sample:", sample)
    return {
        "stm_results": list_conversation
    }
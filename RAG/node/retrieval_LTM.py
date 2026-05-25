from RAG.state import State

def retrieval_LSM(state: State):
    long_term_memory = state["ltm"]
    user_prompt = state["user_prompt"]
    # Lấy tất cả conversation trong LTM
    list_conversation = long_term_memory.query(query_text= user_prompt)
    # list[Dict]
    print("[retrieval_LTM] count:", 0 if list_conversation is None else len(list_conversation))
    if list_conversation:
        sample = list_conversation[:3]
        print("[retrieval_LTM] sample:", sample)
    return {
        "ltm_results": list_conversation
    }
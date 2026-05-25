from ddgs import DDGS
from RAG.state import State

def search(state: State):
    user_prompt = state['user_prompt']
    with DDGS() as d:
        results = list(d.text(user_prompt, max_results=5))
    # Chỉ lấy nội dung
    context = [r['body'] for r in results]
    print("[search] count:", 0 if context is None else len(context))
    if context:
        sample = context[:3]
        print("[search] sample:", sample)
    return {
        "search_results": context,
        "need_research": True
    }

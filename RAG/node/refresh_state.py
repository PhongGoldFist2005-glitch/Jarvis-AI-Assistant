from RAG.state import State


def refresh_state(state: State):
	return {
		"user_prompt": None,
		"stm_results": None,
		"ltm_results": None,
		"search_results": None,
		"summary": None,
		"response": None,
		"context": None,
		"need_research": False,
		"need_old_conversation": False,
		"relevant_personal_info": False,
		"retrieval_stm_done": False,
		"retrieval_ltm_done": False,
		"research_done": False,
	}

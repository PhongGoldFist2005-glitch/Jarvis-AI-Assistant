from langgraph.graph import END, StateGraph

try:
	from langgraph.checkpoint.memory import MemorySaver
	from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
except Exception:
	MemorySaver = None
	JsonPlusSerializer = None

from RAG.node.cache import adding_cache, processing_text
from RAG.node.LLM_gen import LLM_gen
from RAG.node.search import search
from state import State
from RAG.node.enough import check_enough_info
from RAG.node.needPre import need_previous
from RAG.node.need_research import need_research
from RAG.node.need_updateLTM import need_update_LTM
from RAG.node.planner import planning
from RAG.node.refresh_state import refresh_state
from RAG.node.remove_dup import remove_duplicate
from RAG.node.retrieval import retrieval
from RAG.node.retrieval_LTM import retrieval_LSM
from RAG.node.retrievel_STM import retrieval_STM
from RAG.node.updateLTM import update_LTM
from RAG.node.updateSTM import update_STM



def _pass_state(_: State):
	return {}


def _mark_retrieval_stm_done(_: State):
	return {"retrieval_stm_done": True}


def _mark_retrieval_ltm_done(_: State):
	return {"retrieval_ltm_done": True}


def _mark_research_done(_: State):
	return {"research_done": True}


def _route_sync_remove_dup(state: State):
	need_retrieval = bool(state.get("need_old_conversation"))
	need_research = bool(state.get("need_research"))

	retrieval_ready = (not need_retrieval) or (
		state.get("retrieval_stm_done") and state.get("retrieval_ltm_done")
	)
	research_ready = (not need_research) or state.get("research_done")

	if retrieval_ready and research_ready:
		return "Remove_duplicate"
	return END


def build_graph():
	graph = StateGraph(State)

	# Dang ky cac node theo dung khoi chuc nang
	graph.add_node("Preprocessing_text", processing_text)
	graph.add_node("Adding_cache", adding_cache)
	graph.add_node("Planner", planning)
	graph.add_node("Need_previous", _pass_state)
	graph.add_node("Retrieval", retrieval)
	graph.add_node("Retrieval_STM", retrieval_STM)
	graph.add_node("Retrieval_LTM", retrieval_LSM)
	graph.add_node("Need_research", _pass_state)
	graph.add_node("Research", search)
	graph.add_node("Mark_retrieval_stm_done", _mark_retrieval_stm_done)
	graph.add_node("Mark_retrieval_ltm_done", _mark_retrieval_ltm_done)
	graph.add_node("Mark_research_done", _mark_research_done)
	graph.add_node("Sync_remove_dup", _pass_state)
	graph.add_node("Remove_duplicate", remove_duplicate)
	graph.add_node("Enough", _pass_state)
	graph.add_node("LLM_gereration", LLM_gen)
	graph.add_node("Update_STM", update_STM)
	graph.add_node("Need_update_LTM", _pass_state)
	graph.add_node("Update_LTM", update_LTM)
	graph.add_node("Refresh_state", refresh_state)

	# Chuyen tu Start -> Preprocessing_text
	graph.set_entry_point("Preprocessing_text")
	# Chuyen tu Preprocessing_text -> Adding_cache
	graph.add_edge("Preprocessing_text", "Adding_cache")
	# Chuyen tu Adding_cache -> Planner
	graph.add_edge("Adding_cache", "Planner")
	
	# Chuyen tu Planner -> Need_previous
	graph.add_edge("Planner", "Need_previous")
	# Chuyen tu Planner -> Need_research
	graph.add_edge("Planner", "Need_research")
	# Chuyen tu Planner -> Enough
	graph.add_edge("Planner", "Enough")

	# Nhanh nhu cau thong tin cu: co -> Retrieval, khong -> END
	graph.add_conditional_edges(
		"Need_previous",
		need_previous,
		{
			"Retrieval": "Retrieval",
			END: END,
		},
	)

	# Chuyen tu Retrieval -> Retrieval_STM va Retrieval_LTM
	graph.add_edge("Retrieval", "Retrieval_STM")
	graph.add_edge("Retrieval", "Retrieval_LTM")
	# Danh dau ket qua STM/LTM -> Sync_remove_dup
	graph.add_edge("Retrieval_STM", "Mark_retrieval_stm_done")
	graph.add_edge("Retrieval_LTM", "Mark_retrieval_ltm_done")
	graph.add_edge("Mark_retrieval_stm_done", "Sync_remove_dup")
	graph.add_edge("Mark_retrieval_ltm_done", "Sync_remove_dup")

	# Nhanh nhu cau research: co -> Research, khong -> END
	graph.add_conditional_edges(
		"Need_research",
		need_research,
		{
			"Research": "Research",
			END: END,
		},
	)

	# Chuyen tu Research -> Mark_research_done -> Sync_remove_dup
	graph.add_edge("Research", "Mark_research_done")
	graph.add_edge("Mark_research_done", "Sync_remove_dup")

	# Du lieu da du -> LLM_gereration, neu khong -> ket thuc
	graph.add_conditional_edges(
		"Enough",
		check_enough_info,
		{
			"LLM_gereration": "LLM_gereration",
			END: END,
		},
	)

	# Dong bo nhanh Retrieval/Research -> Remove_duplicate
	graph.add_conditional_edges(
		"Sync_remove_dup",
		_route_sync_remove_dup,
		{
			"Remove_duplicate": "Remove_duplicate",
			END: END,
		},
	)

	# Chuyen tu Remove_duplicate -> LLM_gereration
	graph.add_edge("Remove_duplicate", "LLM_gereration")

	# Chuyen tu LLM_gereration -> Update_STM
	graph.add_edge("LLM_gereration", "Update_STM")

	# Chuyen tu Update_STM -> Need_update_LTM
	graph.add_edge("Update_STM", "Need_update_LTM")

	# Kiem tra update LTM: co -> Update_LTM, khong -> Refresh_state
	graph.add_conditional_edges(
		"Need_update_LTM",
		need_update_LTM,
		{
			"Update_LTM": "Update_LTM",
			END: "Refresh_state",
		},
	)

	# Chuyen tu Update_LTM -> Refresh_state
	graph.add_edge("Update_LTM", "Refresh_state")
	# Chuyen tu Refresh_state -> End
	graph.add_edge("Refresh_state", END)

	return graph.compile()


def get_graph():
	return build_graph()

import json
import warnings

try:
	from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
except Exception:
	class LangChainPendingDeprecationWarning(Warning):
		pass

# Tat canh bao allowed_objects cua LangGraph serializer truoc khi import Graph
warnings.filterwarnings(
	"ignore",
	category=LangChainPendingDeprecationWarning,
	message=".*allowed_objects.*",
)
warnings.filterwarnings(
	"ignore",
	category=LangChainPendingDeprecationWarning,
	module="langgraph\\.cache\\.base",
)

from RAG.graph import get_graph
from RAG.memory.LTM import Long_Term_Memory
from RAG.memory.STM import Short_Term_Memory


def _load_reversed_jsonl(jsonl_path: str) -> dict:
	# Doc tung dong JSONL va gom tat ca key/value vao 1 dict duy nhat
	mapping: dict = {}
	with open(jsonl_path, "r", encoding="utf-8") as file:
		for line in file:
			text = line.strip()
			if not text:
				continue
			try:
				item = json.loads(text)
			except json.JSONDecodeError:
				continue

			if isinstance(item, dict):
				# Neu la dict co 1 cap key/value duy nhat thi merge truc tiep
				if len(item) == 1:
					mapping.update(item)
					continue
				# Neu co cap key/value ro rang thi lay theo cap do
				if "key" in item and "value" in item:
					mapping[item["key"]] = item["value"]
					continue

	return mapping


def build_demo_state(user_prompt: str, input_json: dict):
	stm = Short_Term_Memory()
	ltm = Long_Term_Memory()

	return {
		"user_prompt": user_prompt,
		"stm": stm,
		"ltm": ltm,
		"stm_results": None,
		"ltm_results": None,
		"search_results": None,
		"summary": None,
		"response": None,
		"context": None,
		"input_json": input_json,
		"need_research": False,
		"need_old_conversation": False,
		"relevant_personal_info": False,
		"retrieval_stm_done": False,
		"retrieval_ltm_done": False,
		"research_done": False,
	}


def run_demo():
	# Demo dau vao
	question = "What is the capital of France?"
	jsonl_path = r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\Database\reversed_word.jsonl"
	input_json = _load_reversed_jsonl(jsonl_path)
	graph = get_graph()
	state = build_demo_state(question, input_json)

	last_state = None
	last_response_state = None
	try: # stream_mode="values"
		for step_state in graph.stream(state, stream_mode="updates"):
			last_state = step_state
			print("Step state update:", step_state)
			if "LLM_gereration" in step_state:
				print("okok")
				result = step_state["LLM_gereration"].get("response")
				print("LLM response update:", result)
			if step_state.get("response") is not None:
				last_response_state = step_state
		result = last_response_state or last_state
		if result is None:
			result = graph.invoke(state)
	except Exception as exc:
		print("Graph bi loi khi chay demo:", exc)
		return

	print("=== Demo Result ===")
	print("User prompt:", result.get("user_prompt"))
	print("Need research:", result.get("need_research"))
	print("Need old conversation:", result.get("need_old_conversation"))
	print("Response:", result.get("response"))
	print("Summary:", result.get("summary"))

	if not result.get("response"):
		print("Luu y: response trong. Neu chay that, can cau hinh API key cho LLM.")


if __name__ == "__main__":
	run_demo()

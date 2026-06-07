import os
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core import prompts
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from RAG.state import State

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenSaving import tokenizer


class LLMOutput(BaseModel):
	output_str: str
	output_summary: str


def _build_prompt(context: str | None, question: str):
	parser = PydanticOutputParser(pydantic_object=LLMOutput)
	format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

	system_prompt = """
	Bạn là "Giã Vịt" — một trợ lý AI thông minh, chính xác, chuyên nghiệp.

	Nhiệm vụ:
	- Trả lời câu hỏi người dùng → output_str
	- Tạo bản tóm tắt → output_summary

	QUY TẮC QUAN TRỌNG:
	- Nếu output_str NGẮN (<= 20 từ) → output_summary = output_str
	- Nếu output_str DÀI → tóm tắt ngắn gọn, giữ ý chính (<= 40 từ)
	- output_summary phải chứa: câu hỏi người dùng, nội dung trả lời, và các thông tin bổ sung liên quan đến câu trả lời của bạn mà người dùng cung cấp (nếu có)

	Phong cách:
	- Ngắn gọn, rõ ràng, không lan man

	RÀNG BUỘC:
	- CHỈ trả về JSON
	- KHÔNG thêm text ngoài JSON

	Ngữ cảnh (nếu có):
	{context}

	Format trả về:
	""" + format_instructions

	prompt = prompts.ChatPromptTemplate.from_messages(
		[
			("system", system_prompt),
			("human", "Yêu cầu của người dùng: {question}"),
		]
	)

	if context is not None:
		prompt = prompt.format_messages(question=question, context=context)
	else:
		prompt = prompt.format_messages(question=question, context="")

	return prompt, parser


def LLM_gen(state: State):
	user_prompt = state.get("user_prompt")
	context = state.get("context")
	if not user_prompt:
		return {"response": None}

	token_ob = tokenizer()
	llm = ChatGoogleGenerativeAI(
		model= token_ob.model_name,
		google_api_key= token_ob.gemini_token,
	)

	prompt, parser = _build_prompt(context, user_prompt)

	try:
		response = llm.invoke(prompt)
		parsed = parser.parse(response.content)
		return {
			"response": parsed.output_str,
			"summary": parsed.output_summary,
		}
	except Exception as exc:
		return {"response": None}

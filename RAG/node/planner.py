import os
import sys
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core import prompts
from langchain_core.output_parsers import PydanticOutputParser
from RAG.state import State
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenSaving import tokenizer

class PlanningOutput(BaseModel):
    need_research: bool
    need_old_conversation: bool
    relevant_personal_info: bool
    fixed_vocal: str

def planning(state: State):
    input_prompt = state["user_prompt"]
    context = state["context"]

    # Rule-based override for realtime queries
    keyword_text = (input_prompt or "").lower()
    realtime_keywords = [
        "thời tiết",
        "hom nay",
        "hôm nay",
        "moi nhat",
        "mới nhất",
        "hien tai",
        "hiện tại",
        "tin tuc",
        "tin tức",
        "ty gia",
        "tỷ giá",
        "gia vang",
        "giá vàng",
        "giao thong",
        "giao thông",
    ]
    force_research = any(key in keyword_text for key in realtime_keywords)

    parser = PydanticOutputParser(pydantic_object=PlanningOutput)
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    system_prompt = """
    Bạn là "Giã Vịt" — một trợ lý AI thông minh, chính xác, và chuyên nghiệp.

    Nhiệm vụ:
    - Xác định liệu có cần phải sửa lỗi chính tả cho câu hỏi của người dùng hay không.
    - Quyết định có cần nghiên cứu thông tin ở ngoài không
    - Xác định có cần thông tin từ cuộc trò chuyện trước đó không
    - Xác định câu hỏi có liên quan đến thông tin cá nhân hay không

    Định nghĩa:
    - fixed_vocal: là câu hỏi của người dùng sau khi đã được sửa lỗi chính tả hoặc lỗi ngữ pháp, nếu không có lỗi thì giữ nguyên câu hỏi gốc
    - need_research = true nếu cần dữ liệu ngoài hoặc kiến thức cập nhật
    - need_old_conversation = true nếu câu hỏi cần bối cảnh từ cuộc hội thoại trước đó
    - Nếu câu hỏi liên quan đến tính cá nhân mà không trả lời được thì need_old_conversation = true
    - Nếu câu hỏi liên quan đến tính cá nhân mà vẫn tự trả lời được và không cần bối cảnh thì need_old_conversation = false
    - relevant_personal_info = true nếu câu hỏi liên quan đến thông tin cá nhân của người dùng

    Quy tắc nhận biết cần research (ưu tiên cao):
    - Câu hỏi về thông tin thời gian thực hoặc thay đổi theo thời gian (ví dụ: thời tiết hôm nay, tin tức mới nhất, tỷ giá, giá vàng, tình hình giao thông)
    - Câu hỏi yêu cầu dữ liệu ngoài hệ thống hoặc cần tra cứu nguồn hiện tại
    - Nếu có các từ khóa như: "thời tiết", "hôm nay", "mới nhất", "hiện tại", "tỷ giá", "giá", "tin tức" thì ưu tiên need_research = true

    Lưu ý:
    - Bạn là trợ lý realtime, ưu tiên research khi có nghi ngờ thiếu dữ liệu cập nhật

    Ràng buộc:
    - CHỈ trả về JSON đúng theo format
    - KHÔNG thêm text ngoài JSON

    Ngữ cảnh (nếu có):
    {context}

    Format trả về:
    """ + format_instructions

    prompt = prompts.ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Yêu cầu của người dùng: {question}")
            ]
    )
    if context is not None:
        prompt = prompt.format_messages(question=input_prompt, context= context)
    else:
        prompt = prompt.format_messages(question=input_prompt)

    token_ob = tokenizer()
    llm = ChatGoogleGenerativeAI(
        model=token_ob.model_name,
        google_api_key=token_ob.gemini_token
    )

    try:
        response = llm.invoke(prompt)
        parsed = parser.parse(response.content)
        user_prompt = parsed.fixed_vocal if parsed.fixed_vocal is not None else input_prompt
        answerable = (not parsed.need_old_conversation) and (not parsed.relevant_personal_info)

        if answerable:
            need_old_conversation = False
            need_research = False
            context_out = context
        else:
            need_old_conversation = parsed.need_old_conversation
            need_research = parsed.need_research
            context_out = None

        if force_research:
            need_research = True

        return {
            "need_research": need_research,
            "need_old_conversation": need_old_conversation,
            "relevant_personal_info": parsed.relevant_personal_info,
            "user_prompt": user_prompt,
            "context": context_out
        }
    except Exception:
        return {
            "need_research": False,
            "need_old_conversation": False,
            "relevant_personal_info": False,
            "user_prompt": input_prompt,
            "context": None
        }
    
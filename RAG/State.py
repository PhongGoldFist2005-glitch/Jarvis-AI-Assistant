from typing import TypedDict
from STM import Short_Term_Memory
from LTM import Long_Term_Memory

class State(TypedDict):
    # Giữ nguyên qua các bước, chỉ bị update khi bị lỗi chính tả hoặc lỗi ngữ pháp
    # Bị restart mỗi khi xong 1 task
    user_prompt: str | None
    # Chỉ bị update khi có đầy đủ thông tin từ cả user và response
    # Chỉ có thể concat không bị refresh
    stm: Short_Term_Memory
    # Chỉ bị update khi có đầy đủ thông tin từ cả user và response
    # Chỉ có thể concat không bị refresh
    ltm: Long_Term_Memory
    # Chỉ bị update khi có thông tin truy vấn mới từ STM
    # Bị restart mỗi khi xong 1 task
    stm_results: list | None
    # Chỉ bị update khi có thông tin truy vấn mới từ LTM
    # Bị restart mỗi khi xong 1 task
    ltm_results: list | None
    # Chỉ bị update khi có kết quả tìm kiếm mới
    # Bị restart mỗi khi xong 1 task
    search_results: list | None
    # Chỉ bị update khi có response mới từ LLM
    # Bị restart mỗi khi xong 1 task
    summary: str | None
    # Chỉ bị update khi có response mới từ LLM
    # Bị restart mỗi khi xong 1 task
    response: str | None
    # Bị update sau mỗi bước, được refresh nếu bước vào bước RAG
    # Bị restart mỗi khi xong 1 task
    context: str | None
    # Giữ nguyên qua các bước
    # Giữ nguyên cố định qua các bước
    input_json: list
    # Bị restart mỗi khi xong 1 task
    need_research: bool
    # Bị restart mỗi khi xong 1 task
    need_old_conversation: bool
    # Bị restart mỗi khi xong 1 task
    relevant_personal_info: bool
    # Co dinh cho dong bo nhanh Retrieval/Research
    retrieval_stm_done: bool
    retrieval_ltm_done: bool
    research_done: bool
    # Không cần biến enough vì chỉ cần biết need_old_conversation và need_research là False là đủ
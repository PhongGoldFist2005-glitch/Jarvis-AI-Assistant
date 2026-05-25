from RAG.state import State
from rapidfuzz import fuzz

# Context cập nhật ở đây có thể không phải là context dùng để truy vấn lần cuối
# input_json: là 1 dict có key và phonetic và value là tên gốc
def processing_text(state: State):
    WINDOW_SIZE_LIST = [4,5,6,7]
    input_text = state["user_prompt"]
    input_json = state["input_json"]
    text_array = input_text.split()
    
    bucket = []
    for window_size in WINDOW_SIZE_LIST:
        for i in range(0, len(text_array), 1):
            if i + window_size > len(text_array):
                break
            text = " ".join(text_array[i:i+window_size])
            bucket.append(text)
    
    max_similarity = float('-inf')
    best_result = None
    best_text = None
    for text in bucket:
        for keys in input_json:
            similarity = fuzz.ratio(text, keys.strip())
            if similarity > max_similarity:
                max_similarity = similarity
                best_text = keys.strip()
    
    if best_text is not None:
        best_result = f"""Nếu trong câu hỏi của người dùng có lỗi chính tả về các tên riêng thì hãy thay nó bằng cái này:{input_json[best_text]}
        . Nếu không có lỗi chính tả nào thì hãy bỏ qua phần này."""
    
    return {
        "context": best_result
    }

def adding_cache(state: State):
    short_term_memory = state["stm"]
    context = state["context"]
    list_cache = short_term_memory.query_window(query_type= 2)

    if list_cache is not None:
        # Gộp các câu trong list cache thành một chuỗi duy nhất
        full_content = " ".join([item["message"] for item in list_cache])
        full_content = f"""Đây là nội dung từ các cuộc trò chuyện gần đây: {full_content}.
            Nếu có thông tin nào trong đó liên quan đến câu hỏi của người dùng thì hãy sử dụng nó để trả lời câu hỏi của người dùng.
            Nếu không có thông tin nào liên quan thì hãy bỏ qua phần này."""
        if context is not None:
            context = full_content + " " + context
        else:
            context = full_content
    
    return {
        "context": context
    }
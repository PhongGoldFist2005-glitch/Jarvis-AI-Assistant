# Query embedded words in ChromaDB
# Class truy vấn:
# Tạo 1 client kết nối với chromaDB đã được nhúng dữ liệu
# Cơ chế để kiểm tra có từ bị sai chính tả không
#   -> Sử dụng mô hình
# Cơ chế để truy vấn từ trong ChromaDB
# Thay thế từ bị sai chính tả bằng từ đúng chính tả nếu có
# Trả về kết quả truy vấn
import chromadb
from modelAssist import modelAssistant
import json

class querryEmbedWord:
    def __init__(self, databasePath, inputJSON, modelObject):
        self.__client = chromadb.Client()
        self.__client = chromadb.PersistentClient(path=databasePath)
        self.__collection = self.__client.get_collection("words")
        # Khởi tạo mô hình kiểm tra chính tả ở đây
        self.__spellChecker = modelObject
        self.__inputJSON = inputJSON
    
    def query(self, queryText, nResults):
        # Kiểm tra chính tả của queryText ở đây
        prompts = f"""
            <prompt>
                <task>Vietnamese spelling error detection</task>

                <instruction>
                    Nhiệm vụ của bạn là phát hiện CỤM TỪ bị sai chính tả trong câu tiếng Việt.

                    Quy tắc:
                    - Chỉ trả về CHÍNH XÁC cụm từ bị sai chính tả.
                    - Không giải thích.
                    - Không thêm ký tự dư thừa.
                    - Nếu không có lỗi, trả về: None
                    - Chỉ được trả về 1 cụm từ sai duy nhất (cụm sai rõ ràng nhất).

                    Lưu ý:
                    - Không sửa câu.
                    - Không trả về câu đầy đủ.
                    - Không trả về nhiều kết quả.
                </instruction>

                <examples>
                    <example>
                        <input>Xin chào Mes ssii</input>
                        <output>Mes ssii</output>
                    </example>
                    <example>
                        <input>Tôi đang học lập trình</input>
                        <output>None</output>
                    </example>
                    <example>
                        <input>mes ba sút bóng</input>
                        <output>mes</output>
                    </example>
                </examples>

                <input>
                    {queryText}
                </input>
            </prompt>
            """
        resultSpellCheck = self.__spellChecker.useModel(prompts)
        print(f"resultSpellCheck: {resultSpellCheck}")
        if resultSpellCheck != "None":
            # Nếu có từ sai, truy vấn từ đúng chính tả trong ChromaDB và thay thế
            fixedWord = self.__collection.query(
                query_texts=[resultSpellCheck],
                n_results=nResults
            )["documents"][0][0]

            fixedWord = self.__inputJSON.get(fixedWord)
            outputString = queryText.replace(resultSpellCheck, fixedWord)
            return outputString
        else:
            return queryText

if __name__ == "__main__":
    # Khởi tạo mô hình
    # Cho mô hình vào class truy vấn
    # Thử truy vấn
    dictResult = {}
    with open(r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\Database\reversed_word.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            dictResult.update(data)

    model = modelAssistant()
    query = querryEmbedWord("./Database/chroma_db", dictResult, model)
    result = query.query("laioneometsi sút bóng", 1)
    print(result)
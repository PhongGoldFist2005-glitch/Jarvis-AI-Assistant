# Quy trình nhúng:
# Tạo 1 collection trong ChromaDB để lưu trữ các từ đã được nhúng.
# Đọc dữ liệu từ file word.jsonl
# Đưa vào collection
import chromadb
import json
from tqdm import tqdm
# "./Database/chroma_db"
# r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\word.jsonl"

def embedWord(databasePath, filePath, lenJSONL):
    client = chromadb.Client()
    client = chromadb.PersistentClient(path=databasePath)
    collection = client.get_or_create_collection("words")

    ids = []
    count = 0
    with open(filePath, "r") as f:
        for tqdmLine in tqdm(f, desc="Embedding words", total= lenJSONL):
            data = json.loads(tqdmLine)
            data = dict(data)
            ids.clear()
            for key in data:
                documents = data[key]
                for i in range(len(documents)):
                    ids.append(f"{count}")
                    count += 1
                collection.add(
                    ids= ids,
                    documents= documents
                )
    return collection

if __name__ == "__main__":
    databasePath = r"./Database/chroma_db"
    filePath = r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\word.jsonl"
    lenJSONL = 495
    collection = embedWord(databasePath, filePath, lenJSONL)
    result = collection.query(
        query_texts=["co co chan no"],
        n_results=3
    )
    print(result)
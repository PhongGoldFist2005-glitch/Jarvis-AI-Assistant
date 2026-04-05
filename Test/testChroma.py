import chromadb
import json

client = chromadb.Client()

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("words")

ids = []
count = 0
with open(r"P:\Program Files\Python313\AI_assistance\Model\word.jsonl","r") as f:
    for line in f:
        data = json.loads(line)
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

result = collection.query(
    query_texts=["co co chan no"],
    n_results=3
)
print(result)
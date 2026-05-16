#  Bạn có biết lai o ne met si không
from rapidfuzz import fuzz
import json

input_text = input("Prompt: ")
WINDOW_SIZE = 5
text_array = input_text.split()
bucket = []
for i in range(0, len(text_array), 1):
    if i + WINDOW_SIZE > len(text_array):
        break
    text = " ".join(text_array[i:i+WINDOW_SIZE])
    bucket.append(text)
    print(text)
all_values = []

with open(
    r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\Database\word.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        data = json.loads(line)

        for value_list in data.values():
            all_values.extend(value_list)

max_similarity = float('-inf')
best_result = None
best_text = None
for text in bucket:
    for line in all_values:
        similarity = fuzz.ratio(text, line.strip())
        # print(f"Similar text: {line.strip()} (Similarity: {similarity}%)")
        if similarity > max_similarity:
            max_similarity = similarity
            best_result = line.strip()
            best_text = text
print(f"Best result: {best_result} (Similarity: {max_similarity}%)")
print(f"Best text: {best_text}")
# Chiều: Lấy đa dạng window size
# Truy vấn được ra text đúng
# Cache jsonl sẽ được truyền vào function cache
# Triển khai langchain cho cơ chế cache
# Tạo state
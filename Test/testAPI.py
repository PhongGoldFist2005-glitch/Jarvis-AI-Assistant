import ollama

client = ollama.Client()

model = "qwen2.5"
message = "Thủ đô của Việt Nam là gì ?. Bạn phải trả lời hoàn toàn bằng Tiếng Việt"

response = client.generate(
    model=model,
    prompt= message,
    options={
        "num_gpu": 1
    })

print("Message from OLLAMA")
print(response["response"])
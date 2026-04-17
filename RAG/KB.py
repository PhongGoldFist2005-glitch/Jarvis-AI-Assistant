from vectorDB import VectorDB

# KB sẽ là các chứa cac thông tin được lưu trữ từ nguồn bên ngoài
# Sẽ được clean sau 1 khoảng thời gian nhất định để việc lưu trữ các thông tin cơ bản cho long term mem được hiệu quả hơn
# Search -> Clean -> Chunking -> Vectorize -> Store
# Viết cơ chế check RAG
# Viết cơ chế cho pipeline search
# Viết class KB
# Viết class ranking
class KB(VectorDB):
    def __init__(self):
        pass
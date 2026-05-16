from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
from param import Param


class Long_Term_Memory:
    """
    Long-term memory system sử dụng Qdrant để lưu trữ thông tin mang tính cá nhân hóa, mang tính chính xác lâu dài.
    """
    def __init__(self):
        """
        Khởi tạo Long-term Memory sử dụng cấu hình từ Param.
        """
        self.host = Param.QDRANT_HOST
        self.collection_name = Param.LTM_COLLECTION_NAME
        self.vector_size = Param.VECTOR_SIZE
        self.model_name = Param.EMBEDDING_MODEL
        
        # Kết nối tới Qdrant
        self.client = QdrantClient(self.host)
        
        # Tải model embedding
        self.model = SentenceTransformer(self.model_name)
        
        # Khởi tạo collection nếu chưa tồn tại
        self._initialize_collection()
        
        # Lưu trữ thông tin truy vấn cuối cùng
        self.last_query_result = []

    def _initialize_collection(self):
        """Khởi tạo collection trong Qdrant nếu chưa tồn tại."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created successfully.")

    def should_save_to_ltm(self, text: str) -> bool:
        # Kiểm tra nội dung không rỗng
        if not text or len(text.strip()) == 0:
            return False
        
        # Kiểm tra chiều dài tối thiểu
        if len(text) < 5:
            return False
        
        return True

    def insert(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        # Kiểm tra xem có nên lưu thông tin này không
        if not self.should_save_to_ltm(text):
            print(f"Thông tin không thỏa điều kiện để lưu vào LTM: '{text[:50]}...'")
            return None

        # Tạo embedding
        embedding = self.model.encode(text, normalize_embeddings=True)

        # Tạo ID duy nhất
        point_id = str(uuid.uuid4()).replace('-', '')[:20]
        point_id = int(point_id, 16) % (2**63)  # Chuyển thành số nguyên hợp lệ

        # Chuẩn bị payload
        payload = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        # Upsert vào Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            ]
        )

        print(f"Thông tin được lưu thành công (ID: {point_id})")
        return str(point_id)

    def query(self, query_text: str, limit: int = 5, score_threshold: float = None) -> List[Dict[str, Any]]:
        if score_threshold is None:
            score_threshold = Param.DEFAULT_SIMILARITY_THRESHOLD
            
        # Tạo embedding cho query
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)

        # Tìm kiếm trong Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )

        # Chuẩn bị kết quả
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text"),
            })

        # Lưu kết quả truy vấn cuối cùng
        self.last_query_result = results

        return results

    def delete(self, query_text: str, similarity_threshold: float = None) -> List[int]:
        if similarity_threshold is None:
            similarity_threshold = Param.DELETE_SIMILARITY_THRESHOLD
            
        # Tìm thông tin liên quan
        related_info = self.query(query_text, limit=100, score_threshold=similarity_threshold)

        # Xóa các thông tin tìm được
        deleted_ids = []
        for info in related_info:
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=[info["id"]]
                )
                deleted_ids.append(info["id"])
                print(f"Đã xóa: {info['text'][:50]}... (ID: {info['id']})")
            except Exception as e:
                print(f"Lỗi khi xóa ID {info['id']}: {e}")

        if deleted_ids:
            print(f"Tổng cộng đã xóa {len(deleted_ids)} thông tin")
        else:
            print("Không tìm thấy thông tin liên quan để xóa")

        return deleted_ids

    def get_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            # Lấy tất cả points từ collection
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit
            )

            results = []
            for point in points:
                results.append({
                    "id": point.id,
                    "text": point.payload.get("text"),
                    "timestamp": point.payload.get("timestamp"),
                    "metadata": point.payload.get("metadata", {})
                })

            return results
        except Exception as e:
            print(f"Lỗi khi lấy tất cả thông tin: {e}")
            return []


if __name__ == "__main__":
    # Test LTM class
    ltm = Long_Term_Memory()

    # Test insert
    print("\n=== Test Insert ===")
    ltm.insert("Tôi thích nghe nhạc pop")
    ltm.insert("Tôi thường làm việc vào buổi sáng")
    ltm.insert("Tôi sống ở Hà Nội")

    # Test query
    print("\n=== Test Query ===")
    results = ltm.query("Sở thích âm nhạc")
    for result in results:
        print(f"- {result['text']} (Score: {result['score']:.2f})")

    # Test get_all
    print("\n=== Test Get All ===")
    all_info = ltm.get_all()
    print(f"Tổng thông tin lưu trữ: {len(all_info)}")

    # Test delete
    print("\n=== Test Delete ===")
    deleted = ltm.delete("xóa thông tin về nhạc")
    
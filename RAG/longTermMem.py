from vectorDB import VectorDB
from typing import Any, List, Dict, Optional
import time
import json
import os


class LongTermMem(VectorDB):
    """
    Lớp LongTermMem kế thừa từ VectorDB với cơ chế quản lý dung lượng.
    Tự động xóa những documents ít liên quan nhất khi số lượng vượt quá top_k.
    Sử dụng SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") cho embedding.
    """
    
    def __init__(self, input_queue=None, top_k: int = 100, persist_directory: str = "./LongTermMem_data",
                model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Khởi tạo LongTermMem.
        
        Args:
            input_queue: Queue đầu vào cho các messages
            top_k (int): Số lượng documents tối đa được giữ lại
            persist_directory (str): Đường dẫn lưu trữ dữ liệu
            model_name (str): Tên model SentenceTransformer (default: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        """
        super().__init__(persist_directory=persist_directory, model_name=model_name)
        self.queue = input_queue
        self.top_k = top_k
        self.access_timestamps = {}  # Lưu lại lần truy cập cuối cùng của mỗi document
        
        # Riêng BM25 storage path cho LongTermMem
        self.bm25_storage_path = os.path.join(persist_directory, "longtermmem_bm25_storage.json")

    # ==================== OVERRIDE ADD METHODS ====================
    def add(self, data: Dict[str, Any], metadata: Optional[Dict] = None) -> str:
        """
        Thêm một document và kiểm tra nếu vượt quá top_k, xóa những documents ít liên quan.
        """
        doc_id = super().add(data, metadata)
        
        # Cập nhật timestamp truy cập
        if doc_id:
            self.access_timestamps[doc_id] = time.time()
            
        # Kiểm tra và loại bỏ documents ít liên quan nếu vượt quá top_k
        self._enforce_capacity()
        
        return doc_id

    def add_batch(self, data_list: List[Dict[str, Any]], 
                metadata_list: Optional[List[Dict]] = None) -> List[str]:
        """
        Thêm nhiều documents và kiểm tra nếu vượt quá top_k, xóa những documents ít liên quan.
        """
        doc_ids = super().add_batch(data_list, metadata_list)
        
        # Cập nhật timestamps
        current_time = time.time()
        for doc_id in doc_ids:
            if doc_id:
                self.access_timestamps[doc_id] = current_time
        
        # Kiểm tra và loại bỏ
        self._enforce_capacity()
        
        return doc_ids

    # ==================== CAPACITY MANAGEMENT ====================
    def _enforce_capacity(self):
        """
        Kiểm tra nếu collection vượt quá top_k, xóa những documents ít liên quan nhất.
        """
        current_count = self.count()
        
        if current_count > self.top_k:
            excess = current_count - self.top_k
            self._remove_least_relevant(excess)

    def _remove_least_relevant(self, num_to_remove: int):
        """
        Xóa `num_to_remove` documents ít liên quan nhất dựa trên:
        1. Tần suất truy cập (access frequency)
        2. Thời gian truy cập cuối (recency)
        3. Độ liên quan (relevance score)
        
        Args:
            num_to_remove (int): Số documents cần xóa
        """
        try:
            # Lấy tất cả documents hiện có
            all_data = self.collection.get()
            
            if not all_data or not all_data['ids']:
                return
            
            doc_ids = all_data['ids']
            doc_scores = []
            
            # Tính điểm cho từng document dựa trên:
            # - Tần suất truy cập (từ access_timestamps)
            # - Recency (lần truy cập gần đây nhất)
            # - Độ dài/quan trọng của document (metadata hoặc text length)
            
            for i, doc_id in enumerate(doc_ids):
                # Lấy timestamp truy cập
                last_access = self.access_timestamps.get(doc_id, time.time())
                time_diff = time.time() - last_access
                
                # Lấy metadata
                metadata = all_data['metadatas'][i] if all_data['metadatas'] else {}
                importance = metadata.get('importance', 0)  # 0-100
                
                # Tính điểm: documents cũ và ít quan trọng sẽ được xóa trước
                # Score thấp = ít giá trị = xóa trước
                recency_score = 1.0 / (1.0 + time_diff / 3600)  # Giảm theo thời gian (tính bằng giờ)
                importance_score = importance / 100.0 if importance > 0 else 0.5
                
                # Score kết hợp: 60% recency + 40% importance
                combined_score = (recency_score * 0.6) + (importance_score * 0.4)
                
                doc_scores.append((doc_id, combined_score))
            
            # Sắp xếp theo score (thấp nhất trước)
            doc_scores.sort(key=lambda x: x[1])
            
            # Xóa những documents có score thấp nhất
            to_delete = [doc_id for doc_id, score in doc_scores[:num_to_remove]]
            
            if to_delete:
                self.delete_batch(to_delete)
                
                # Xóa khỏi access_timestamps
                for doc_id in to_delete:
                    self.access_timestamps.pop(doc_id, None)
                
                # Xóa khỏi BM25 storage
                self._remove_from_bm25(to_delete)
                
                print(f"[LongTermMem] Đã xóa {len(to_delete)} documents ít liên quan. "
                    f"Số documents còn lại: {self.count()}/{self.top_k}")
        
        except Exception as e:
            print(f"Lỗi khi loại bỏ documents ít liên quan: {e}")

    def _remove_from_bm25(self, doc_ids_to_delete: List[str]) -> None:
        """
        Xóa documents khỏi BM25 storage khi chúng bị xóa từ vector DB.
        
        Args:
            doc_ids_to_delete (List[str]): Danh sách doc_ids cần xóa
        """
        try:
            # Tìm indices của các documents cần xóa trong BM25
            indices_to_keep = []
            for i, bm25_doc_id in enumerate(self.bm25_doc_ids):
                if bm25_doc_id not in doc_ids_to_delete:
                    indices_to_keep.append(i)
            
            # Giữ lại những documents không cần xóa
            new_corpus = [self.bm25_corpus[i] for i in indices_to_keep]
            new_corpus_raw = [self.bm25_corpus_raw[i] for i in indices_to_keep]
            new_doc_ids = [self.bm25_doc_ids[i] for i in indices_to_keep]
            
            self.bm25_corpus = new_corpus
            self.bm25_corpus_raw = new_corpus_raw
            self.bm25_doc_ids = new_doc_ids
            
            # Tái chỉ mục BM25
            if self.bm25_corpus:
                self.bm25_retriever = __import__('bm25s').BM25()
                self.bm25_retriever.index(self.bm25_corpus)
            else:
                self.bm25_retriever = None
            
            # Lưu lại storage
            self._save_bm25_storage()
            
        except Exception as e:
            print(f"⚠ Lỗi xóa từ BM25: {e}")

    # ==================== OVERRIDE RETRIEVE METHODS ====================
    def retrieve(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy document và cập nhật timestamp truy cập.
        """
        result = super().retrieve(object_id)
        
        if result:
            # Cập nhật thời gian truy cập cuối
            self.access_timestamps[object_id] = time.time()
        
        return result

    def retrieve_batch(self, object_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        Lấy nhiều documents và cập nhật timestamp truy cập.
        """
        results = super().retrieve_batch(object_ids)
        
        # Cập nhật thời gian truy cập cuối
        current_time = time.time()
        for doc_id in object_ids:
            self.access_timestamps[doc_id] = current_time
        
        return results

    def query(self, query_text: str, top_k: int = 5, 
            filters: Optional[Dict] = None) -> List[Dict]:
        """
        Truy vấn và cập nhật timestamp truy cập cho các documents được trả về.
        """
        results = super().query(query_text, top_k, filters)
        
        # Cập nhật thời gian truy cập cuối cho các documents được trả về
        current_time = time.time()
        for result in results:
            self.access_timestamps[result['id']] = current_time
        
        return results

    def query_hybrid(self, query_text: str, top_k: int = 5, 
                    semantic_weight: float = 0.6,
                    filters: Optional[Dict] = None) -> List[Dict]:
        """
        Truy vấn hybrid (Semantic + BM25) và cập nhật timestamp truy cập.
        
        Args:
            query_text (str): Văn bản truy vấn
            top_k (int): Số kết quả trả về
            semantic_weight (float): Trọng số cho semantic search (0-1)
            filters (Optional[Dict]): Bộ lọc bổ sung
        
        Returns:
            List[Dict]: Danh sách các kết quả được rerank
        """
        results = super().query_hybrid(query_text, top_k, semantic_weight, filters)
        
        # Cập nhật thời gian truy cập cuối cho các documents được trả về
        current_time = time.time()
        for result in results:
            doc_id = result.get('id') or result.get('doc_id')
            if doc_id:
                self.access_timestamps[doc_id] = current_time
        
        return results

    # ==================== UTILITY ====================
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thông tin thống kê về LongTermMem.
        
        Returns:
            Dict chứa thông tin về số documents, dung lượng, v.v.
        """
        current_count = self.count()
        
        return {
            'current_count': current_count,
            'max_capacity': self.top_k,
            'utilization': f"{(current_count / self.top_k * 100):.1f}%",
            'tracked_documents': len(self.access_timestamps),
            'db_name': self.db_name,
            'persist_directory': self.persist_directory
        }
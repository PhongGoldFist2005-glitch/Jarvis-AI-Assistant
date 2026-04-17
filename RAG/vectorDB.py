import chromadb
from typing import Any, List, Dict, Optional
import uuid
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import json
import os
from pyvi import ViTokenizer
import bm25s
import numpy as np


class VectorDB:
    """
    Base class cho Vector Database sử dụng ChromaDB.
    Cung cấp các phương thức triển khai cơ bản cho các hoạt động: tạo, thêm, truy xuất, xóa.
    Sử dụng SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") cho embedding.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Khởi tạo Vector Database.
        
        Args:
            persist_directory (str): Đường dẫn lưu trữ dữ liệu ChromaDB
            model_name (str): Tên model SentenceTransformer để sử dụng (default: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.db_name = None
        self.model_name = model_name
        
        # Khởi tạo SentenceTransformer embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        # Khởi tạo BM25
        self.bm25_storage_path = os.path.join(persist_directory, "vectordb_bm25_storage.json")
        self.bm25_corpus: List[List[str]] = []  # Danh sách các token sequences
        self.bm25_corpus_raw: List[str] = []  # Danh sách các documents gốc
        self.bm25_retriever = None
        self.bm25_doc_ids: List[str] = []  # Lưu doc_id tương ứng
        self._load_bm25_storage()

    # ==================== BM25 STORAGE METHODS ====================
    def _load_bm25_storage(self) -> None:
        """Tải dữ liệu BM25 từ file JSON"""
        try:
            if os.path.exists(self.bm25_storage_path):
                with open(self.bm25_storage_path, 'r', encoding='utf-8') as f:
                    storage_data = json.load(f)
                    self.bm25_corpus = storage_data.get("corpus", [])
                    self.bm25_corpus_raw = storage_data.get("corpus_raw", [])
                    self.bm25_doc_ids = storage_data.get("doc_ids", [])
                
                if self.bm25_corpus:
                    self.bm25_retriever = bm25s.BM25()
                    self.bm25_retriever.index(self.bm25_corpus)
                    print(f"✓ Đã tải {len(self.bm25_corpus)} chunks BM25 từ {self.bm25_storage_path}")
        except Exception as e:
            print(f"⚠ Lỗi tải BM25 storage: {e}")

    def _save_bm25_storage(self) -> None:
        """Lưu dữ liệu BM25 vào file JSON"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            storage_data = {
                "corpus": self.bm25_corpus,
                "corpus_raw": self.bm25_corpus_raw,
                "doc_ids": self.bm25_doc_ids
            }
            with open(self.bm25_storage_path, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Lỗi lưu BM25 storage: {e}")

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text sử dụng pyvi"""
        try:
            tokenized = ViTokenizer.tokenize(text)
            return tokenized.split()
        except:
            # Fallback: split by space nếu tokenize thất bại
            return text.split()

    def _normalize_bm25_scores(self, scores: np.ndarray) -> np.ndarray:
        """Chuẩn hóa điểm BM25 về khoảng [0, 1] bằng Sigmoid"""
        if len(scores) == 0 or np.all(scores == 0):
            return scores
        
        scores_normalized = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
        sigmoid_scores = 1 / (1 + np.exp(-scores_normalized))
        return sigmoid_scores

    def _add_to_bm25(self, doc_id: str, text: str) -> None:
        """Thêm document vào BM25"""
        try:
            tokens = self._tokenize_text(text)
            self.bm25_corpus.append(tokens)
            self.bm25_corpus_raw.append(text)
            self.bm25_doc_ids.append(doc_id)
            
            # Tái chỉ mục BM25
            if self.bm25_corpus:
                self.bm25_retriever = bm25s.BM25()
                self.bm25_retriever.index(self.bm25_corpus)
            
            self._save_bm25_storage()
        except Exception as e:
            print(f"⚠ Lỗi thêm vào BM25: {e}")

    def _query_bm25(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Truy vấn BM25"""
        if not self.bm25_retriever or len(self.bm25_corpus) == 0:
            return []
        
        try:
            query_tokens = self._tokenize_text(query_text)
            scores = self.bm25_retriever.get_scores(query_tokens)
            
            # Lấy top_k
            top_indices = np.argsort(scores)[::-1][:min(top_k, len(scores))]
            top_scores = scores[top_indices]
            
            # Chuẩn hóa
            if len(top_scores) > 0:
                top_scores = self._normalize_bm25_scores(top_scores)
            
            results = []
            for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
                idx = int(idx)
                if idx < len(self.bm25_doc_ids):
                    results.append({
                        'rank': rank + 1,
                        'doc_id': self.bm25_doc_ids[idx],
                        'text': self.bm25_corpus_raw[idx],
                        'bm25_score': float(score),
                        'index': idx
                    })
            
            return results
        except Exception as e:
            print(f"⚠ Lỗi truy vấn BM25: {e}")
            return []

    def _hybrid_reranking(self, semantic_results: List[Dict], 
                        bm25_results: List[Dict], 
                        semantic_weight: float = 0.6) -> List[Dict]:
        """
        Kết hợp kết quả semantic và BM25
        
        Args:
            semantic_results: kết quả từ semantic search
            bm25_results: kết quả từ BM25
            semantic_weight: trọng số cho semantic search (0-1)
        
        Returns:
            Danh sách kết quả sau khi reranking
        """
        # Tạo dictionary để lưu scores
        doc_scores = {}
        
        # Thêm semantic scores
        for i, result in enumerate(semantic_results):
            doc_id = result.get('id')
            # Normalize semantic distance (0-1, thấp hơn là tốt hơn)
            distance = result.get('distance', 1.0)
            semantic_score = 1 - (distance / 2.0)  # Giả sử distance max là 2
            semantic_score = max(0, min(1, semantic_score))  # Clamp to [0, 1]
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {}
            doc_scores[doc_id]['semantic'] = semantic_score
            doc_scores[doc_id]['data'] = result

        # Thêm BM25 scores
        for result in bm25_results:
            doc_id = result.get('doc_id')
            bm25_score = result.get('bm25_score', 0)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {}
            doc_scores[doc_id]['bm25'] = bm25_score
            doc_scores[doc_id]['bm25_data'] = result

        # Tính hybrid score
        hybrid_results = []
        for doc_id, scores in doc_scores.items():
            semantic_score = scores.get('semantic', 0)
            bm25_score = scores.get('bm25', 0)
            bm25_weight = 1 - semantic_weight
            
            hybrid_score = (semantic_score * semantic_weight) + (bm25_score * bm25_weight)
            
            result_data = scores.get('data', scores.get('bm25_data', {}))
            result_data['hybrid_score'] = hybrid_score
            result_data['semantic_score'] = semantic_score
            result_data['bm25_score'] = bm25_score
            
            hybrid_results.append(result_data)
        
        # Sắp xếp theo hybrid score
        hybrid_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        return hybrid_results
    def create(self, db_name: str, **kwargs) -> bool:
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.db_name = db_name

            # Tạo collection mới hoặc lấy collection nếu đã tồn tại
            # Sử dụng SentenceTransformer embedding function
            # ChromaDB không cho phép metadata rỗng, nên chỉ truyền nếu có
            metadata = kwargs.get("metadata", None)
            if metadata and len(metadata) > 0:
                self.collection = self.client.get_or_create_collection(
                    name=db_name,
                    embedding_function=self.embedding_function,
                    metadata=metadata
                )
            else:
                self.collection = self.client.get_or_create_collection(
                    name=db_name,
                    embedding_function=self.embedding_function
                )

            print(f"Collection '{db_name}' đã sẵn sàng (Model: {self.model_name}).")
            return True

        except Exception as e:
            print(f"Lỗi khi tạo database: {e}")
            self.collection = None
            return False

    def connect(self, db_name: str, **kwargs) -> bool:
        """
        Kết nối đến một collection đã tồn tại trong ChromaDB.
        
        Args:
            db_name (str): Tên của collection
            **kwargs: Các tham số bổ sung
        
        Returns:
            bool: True nếu kết nối thành công
        """
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.db_name = db_name
            # Sử dụng embedding function khi kết nối
            self.collection = self.client.get_or_create_collection(
                name=db_name,
                embedding_function=self.embedding_function
            )
            print(f"Kết nối collection '{db_name}' thành công (Model: {self.model_name}).")
            return True
        except Exception as e:
            print(f"Lỗi khi kết nối database: {e}")
            return False

    # ==================== CREATE/ADD ====================
    def add(self, data: Dict[str, Any], metadata: Optional[Dict] = None) -> str:
        """
        Thêm một document mới vào collection.
        
        Args:
            data (Dict): Dữ liệu document cần thêm (phải chứa 'text' hoặc 'embedding')
            metadata (Optional[Dict]): Metadata của document
        
        Returns:
            str: ID của document vừa thêm
        """
        if self.collection is None:
            print("❌ Lỗi: Collection chưa được tạo. Hãy gọi create() trước.")
            return None
        
        try:
            doc_id = str(uuid.uuid4())
            
            # Lấy text từ data
            text = data.get("text", str(data))
            
            # ChromaDB yêu cầu metadata không rỗng
            if not metadata:
                metadata = {"added": "true"}
            
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata]
            )
            
            # Thêm vào BM25
            self._add_to_bm25(doc_id, text)
            
            return doc_id
        except Exception as e:
            print(f"Lỗi khi thêm document: {e}")
            return None

    def add_batch(self, data_list: List[Dict[str, Any]], 
                metadata_list: Optional[List[Dict]] = None) -> List[str]:
        """
        Thêm nhiều documents cùng lúc.
        
        Args:
            data_list (List[Dict]): Danh sách các documents cần thêm
            metadata_list (Optional[List[Dict]]): Danh sách metadata tương ứng
        
        Returns:
            List[str]: Danh sách IDs của các documents vừa thêm
        """
        if self.collection is None:
            print("❌ Lỗi: Collection chưa được tạo. Hãy gọi create() trước.")
            return []
        
        try:
            ids = [str(uuid.uuid4()) for _ in data_list]
            documents = [d.get("text", str(d)) for d in data_list]
            
            # ChromaDB yêu cầu metadata không rỗng
            if metadata_list:
                metadatas = [{"idx": str(i), **m} if m else {"idx": str(i), "added": "true"} for i, m in enumerate(metadata_list)]
            else:
                metadatas = [{"idx": str(i), "added": "true"} for i in range(len(data_list))]
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            # Thêm vào BM25
            for doc_id, text in zip(ids, documents):
                self._add_to_bm25(doc_id, text)
            
            return ids
        except Exception as e:
            print(f"Lỗi khi thêm batch documents: {e}")
            return []

    # ==================== RETRIEVE/QUERY ====================
    def query(self, query_text: str, top_k: int = 5, 
            filters: Optional[Dict] = None) -> List[Dict]:
        """
        Truy vấn/tìm kiếm trong collection.
        
        Args:
            query_text (str): Văn bản truy vấn
            top_k (int): Số kết quả trả về (mặc định 5)
            filters (Optional[Dict]): Bộ lọc bổ sung
        
        Returns:
            List[Dict]: Danh sách các kết quả tìm được
        """
        if self.collection is None:
            print("❌ Lỗi: Collection chưa được tạo. Hãy gọi create() trước.")
            return []
        
        try:
            where = filters if filters else None
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where
            )
            
            # Format lại kết quả
            formatted_results = []
            if results and results['ids'] and len(results['ids']) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    formatted_results.append({
                        'id': doc_id,
                        'text': results['documents'][0][i] if results['documents'] else '',
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Lỗi khi truy vấn: {e}")
            return []

    def query_hybrid(self, query_text: str, top_k: int = 5, 
                    semantic_weight: float = 0.6,
                    filters: Optional[Dict] = None) -> List[Dict]:
        """
        Truy vấn kết hợp Semantic Search + BM25 với hybrid reranking.
        
        Args:
            query_text (str): Văn bản truy vấn
            top_k (int): Số kết quả trả về
            semantic_weight (float): Trọng số cho semantic search (0-1)
            filters (Optional[Dict]): Bộ lọc bổ sung
        
        Returns:
            List[Dict]: Danh sách các kết quả được rerank theo hybrid score
        """
        try:
            # 1. Truy vấn semantic
            semantic_results = self.query(query_text, top_k * 2, filters)
            
            # 2. Truy vấn BM25
            bm25_results = self._query_bm25(query_text, top_k * 2)
            
            # 3. Kết hợp và rerank
            if not semantic_results and not bm25_results:
                return []
            
            hybrid_results = self._hybrid_reranking(semantic_results, bm25_results, semantic_weight)
            
            # Trả về top_k kết quả
            return hybrid_results[:top_k]
        
        except Exception as e:
            print(f"Lỗi khi truy vấn hybrid: {e}")
            return []

    def retrieve(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy một document cụ thể theo ID.
        
        Args:
            object_id (str): ID của document cần lấy
        
        Returns:
            Optional[Dict]: Document nếu tìm thấy, None nếu không
        """
        if self.collection is None:
            print("❌ Lỗi: Collection chưa được tạo. Hãy gọi create() trước.")
            return None
        
        try:
            result = self.collection.get(ids=[object_id])
            if result and result['ids']:
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0] if result['documents'] else '',
                    'metadata': result['metadatas'][0] if result['metadatas'] else {}
                }
            return None
        except Exception as e:
            print(f"Lỗi khi truy xuất document: {e}")
            return None

    def retrieve_batch(self, object_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        Lấy nhiều documents cùng lúc theo danh sách IDs.
        
        Args:
            object_ids (List[str]): Danh sách IDs cần lấy
        
        Returns:
            List[Optional[Dict]]: Danh sách các documents
        """
        if self.collection is None:
            print("❌ Lỗi: Collection chưa được tạo. Hãy gọi create() trước.")
            return []
        
        try:
            result = self.collection.get(ids=object_ids)
            documents = []
            
            if result and result['ids']:
                for i, doc_id in enumerate(result['ids']):
                    documents.append({
                        'id': doc_id,
                        'text': result['documents'][i] if result['documents'] else '',
                        'metadata': result['metadatas'][i] if result['metadatas'] else {}
                    })
            
            return documents
        except Exception as e:
            print(f"Lỗi khi truy xuất batch documents: {e}")
            return []

    def filter(self, filters: Dict) -> List[Dict[str, Any]]:
        """
        Truy vấn dữ liệu dựa trên điều kiện lọc metadata.
        
        Args:
            filters (Dict): Các điều kiện lọc
        
        Returns:
            List[Dict]: Danh sách các documents thỏa mãn điều kiện
        """
        if self.collection is None:
            print("❌ Lỗi: Collection chưa được tạo. Hãy gọi create() trước.")
            return []
        
        try:
            result = self.collection.get(where=filters)
            documents = []
            
            if result and result['ids']:
                for i, doc_id in enumerate(result['ids']):
                    documents.append({
                        'id': doc_id,
                        'text': result['documents'][i] if result['documents'] else '',
                        'metadata': result['metadatas'][i] if result['metadatas'] else {}
                    })
            
            return documents
        except Exception as e:
            print(f"Lỗi khi lọc documents: {e}")
            return []

    # ==================== UPDATE ====================
    def update(self, object_id: str, data: Dict[str, Any], 
            metadata: Optional[Dict] = None) -> bool:
        """
        Cập nhật một document trong collection.
        
        Args:
            object_id (str): ID của document cần cập nhật
            data (Dict): Dữ liệu mới
            metadata (Optional[Dict]): Metadata mới
        
        Returns:
            bool: True nếu cập nhật thành công
        """
        if self.collection is None:
            print("❌ Lỗi: Collection chưa được tạo. Hãy gọi create() trước.")
            return False
        
        try:
            text = data.get("text", str(data))
            meta = metadata if metadata else {"updated": "true"}
            
            self.collection.update(
                ids=[object_id],
                documents=[text],
                metadatas=[meta]
            )
            return True
        except Exception as e:
            print(f"Lỗi khi cập nhật document: {e}")
            return False

    # ==================== DELETE ====================
    def delete_batch(self, object_ids: List[str]) -> int:
        """
        Xóa nhiều documents cùng lúc.
        
        Args:
            object_ids (List[str]): Danh sách IDs cần xóa
        
        Returns:
            int: Số documents đã xóa thành công
        """
        if self.collection is None:
            return 0
        
        try:
            self.collection.delete(ids=object_ids)
            return len(object_ids)
        except Exception as e:
            print(f"Lỗi khi xóa batch documents: {e}")
            return 0

    def clear(self) -> bool:
        """
        Xóa toàn bộ dữ liệu khỏi collection.
        
        Returns:
            bool: True nếu xóa thành công
        """
        if self.collection is None:
            return False
        
        try:
            if self.collection:
                # Lấy tất cả IDs rồi xóa
                all_data = self.collection.get()
                if all_data and all_data['ids']:
                    self.collection.delete(ids=all_data['ids'])
            return True
        except Exception as e:
            print(f"Lỗi khi xóa toàn bộ dữ liệu: {e}")
            return False

    # ==================== UTILITY ====================
    def count(self) -> int:
        """
        Đếm số lượng documents trong collection.
        
        Returns:
            int: Tổng số documents
        """
        if self.collection is None:
            return 0
        
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Lỗi khi đếm documents: {e}")
            return 0

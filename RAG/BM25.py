import os
import json
from typing import List, Dict, Tuple
from pyvi import ViTokenizer
import bm25s
import numpy as np


class BM25Pipeline:
    """
    Pipeline BM25 cho tiếng Việt với các bước:
    1. Chunking text bằng pyvi
    2. Lưu trữ chunks trong kho
    3. Đánh giá điểm BM25s và chuẩn hóa 0-1
    """
    
    def __init__(self, storage_path: str = None, max_corpus_size: int = 100):
        """
        Khởi tạo BM25Pipeline
        
        Args:
            storage_path: đường dẫn lưu trữ kho chunks (tùy chọn)
            max_corpus_size: kích thước tối đa của kho (default=100), xóa cũ khi vượt
        """
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(__file__), "bm25_storage.json"
        )
        self.max_corpus_size = max_corpus_size
        self.corpus: List[List[str]] = []  # Lưu chunks đã tokenize (list of tokens)
        self.corpus_raw: List[str] = []  # Lưu chunks gốc (string)
        self.retriever = None
        self.load_storage()
    
    def _manage_corpus_size(self) -> None:
        """
        Quản lý kích thước kho, xóa items cũ khi vượt quá max_corpus_size (FIFO)
        """
        while len(self.corpus) > self.max_corpus_size:
            removed_tokens = self.corpus.pop(0)
            removed_text = self.corpus_raw.pop(0)
            print(f"🗑️  Xóa chunk cũ: '{removed_text[:50]}...'")
        
        # Huấn luyện lại retriever nếu cần
        if len(self.corpus) > 0:
            self.retriever = bm25s.BM25()
            self.retriever.index(self.corpus)
    
    def chunk_text(self, text: str, chunk_size: int = 100) -> List[str]:
        """
        Chia text thành các chunks và tokenize bằng pyvi
        
        Args:
            text: text cần chia
            chunk_size: kích thước mỗi chunk (số ký tự)
            
        Returns:
            List các chunks đã tokenize
        """
        # Chia text thành chunks
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        
        # Tokenize từng chunk bằng pyvi
        tokenized_chunks = []
        for chunk in chunks:
            try:
                tokenized = ViTokenizer.tokenize(chunk)
                tokenized_chunks.append(tokenized)
            except Exception as e:
                print(f"Lỗi tokenize: {e}, giữ chunk gốc")
                tokenized_chunks.append(chunk)
        
        return tokenized_chunks
    
    def add_documents(self, documents: List[str], chunk_size: int = 100) -> None:
        """
        Thêm documents vào kho BM25
        
        Args:
            documents: danh sách các document
            chunk_size: kích thước chunk
        """
        for doc in documents:
            chunks = self.chunk_text(doc, chunk_size)
            for chunk in chunks:
                # Lưu raw string
                self.corpus_raw.append(chunk)
                # Tokenize chunk thành list of tokens (split by space)
                tokens = chunk.split()
                self.corpus.append(tokens)
        
        # Huấn luyện lại retriever
        self.retriever = bm25s.BM25()
        self.retriever.index(self.corpus)
        print(f"✓ Đã thêm {len(documents)} documents, tổng chunks: {len(self.corpus)}")
        
        # Quản lý kích thước kho
        self._manage_corpus_size()
    
    def save_storage(self) -> None:
        """Lưu kho chunks vào file JSON"""
        storage_data = {
            "corpus": self.corpus,
            "corpus_raw": self.corpus_raw
        }
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(storage_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Đã lưu kho vào: {self.storage_path}")
    
    def load_storage(self) -> None:
        """Tải kho chunks từ file JSON"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                storage_data = json.load(f)
                # Corpus được lưu là list of list
                self.corpus = storage_data.get("corpus", [])
                self.corpus_raw = storage_data.get("corpus_raw", [])
            if self.corpus:
                self.retriever = bm25s.BM25()
                self.retriever.index(self.corpus)
            print(f"✓ Đã tải {len(self.corpus)} chunks từ kho")
    
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa điểm BM25 về khoảng [0, 1] bằng Sigmoid function
        
        Args:
            scores: mảng điểm BM25
            
        Returns:
            mảng điểm đã chuẩn hóa [0, 1] bằng sigmoid
        """
        if len(scores) == 0:
            return scores
        
        # Sigmoid function: σ(x) = 1 / (1 + e^(-x))
        # Trước tiên chuẩn hóa scores để vào khoảng hợp lý
        scores_normalized = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
        
        # Áp dụng sigmoid
        sigmoid_scores = 1 / (1 + np.exp(-scores_normalized))
        
        return sigmoid_scores
    
    def retrieve(self, query: str, top_k: int = 5, 
                normalize: bool = True) -> List[Dict]:
        """
        Tìm kiếm các chunks tương liên với query
        
        Args:
            query: câu query
            top_k: số kết quả trả về
            normalize: có chuẩn hóa điểm hay không
            
        Returns:
            List các dict chứa:
            - 'text': chunk text
            - 'score': điểm BM25 (0-1 nếu normalize=True)
            - 'index': vị trí chunk trong corpus
        """
        if not self.retriever or len(self.corpus) == 0:
            print("⚠ Kho trống, vui lòng thêm documents trước")
            return []
        
        # Tokenize query
        try:
            tokenized_query = ViTokenizer.tokenize(query)
        except:
            tokenized_query = query
        
        # Convert query string thành list of tokens
        query_tokens = tokenized_query.split()
        
        # Tính scores cho query sử dụng get_scores
        scores = self.retriever.get_scores(query_tokens)
        
        # Lấy top_k results
        top_indices = np.argsort(scores)[::-1][:min(top_k, len(scores))]
        top_scores = scores[top_indices]
        
        # Chuẩn hóa điểm nếu cần
        if normalize and len(top_scores) > 0:
            top_scores = self.normalize_scores(top_scores)
        
        # Định dạng kết quả
        output = []
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
            idx = int(idx)
            output.append({
                'rank': rank + 1,
                'text': self.corpus_raw[idx] if idx < len(self.corpus_raw) else "",
                'score': float(score),
                'index': idx
            })
        
        return output
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> Dict:
        """
        Tìm kiếm hàng loạt queries
        
        Args:
            queries: danh sách queries
            top_k: số kết quả mỗi query
            
        Returns:
            Dict với keys là queries và values là kết quả tìm kiếm
        """
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, top_k)
        return results
    
    def get_corpus_stats(self) -> Dict:
        """Lấy thống kê về kho"""
        return {
            'total_chunks': len(self.corpus),
            'max_corpus_size': self.max_corpus_size,
            'utilization': f"{len(self.corpus) / self.max_corpus_size * 100:.1f}%",
            'avg_chunk_length': np.mean([len(c) for c in self.corpus]) if self.corpus else 0,
            'min_chunk_length': min([len(c) for c in self.corpus]) if self.corpus else 0,
            'max_chunk_length': max([len(c) for c in self.corpus]) if self.corpus else 0,
            'storage_path': self.storage_path
        }
    
    def set_max_corpus_size(self, new_size: int) -> None:
        """
        Thay đổi kích thước tối đa của kho
        
        Args:
            new_size: kích thước tối đa mới
        """
        old_size = self.max_corpus_size
        self.max_corpus_size = new_size
        print(f"✓ Thay đổi max_corpus_size: {old_size} → {new_size}")
        # Quản lý lại nếu kích thước mới nhỏ hơn kích thước hiện tại
        self._manage_corpus_size()


# ============ TEST ============
if __name__ == "__main__":
    # Khởi tạo pipeline với max_corpus_size = 100
    bm25_pipeline = BM25Pipeline(max_corpus_size=100)
    
    # Test documents
    documents = [
        "Hà Nội là thủ đô của Việt Nam. Nó là một trong những thành phố lớn nhất Đông Nam Á.",
        "Sài Gòn là thành phố lớn nhất Việt Nam với dân số hơn 8 triệu người.",
        "Đà Nẵng là thành phố ven biển nổi tiếng với cảnh đẹp và du lịch phát triển.",
        "Hội An là một thành phố cổ nằm ở miền Trung Việt Nam, được UNESCO công nhận là di sản thế giới."
    ]
    
    # Thêm documents vào kho
    bm25_pipeline.add_documents(documents)
    
    # Lưu kho
    bm25_pipeline.save_storage()
    
    # In thống kê kho
    stats = bm25_pipeline.get_corpus_stats()
    print("\n📊 Thống kê kho:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demo quản lý kho: thêm nhiều documents để vượt quá max_corpus_size
    print("\n📦 Demo quản lý kho (thêm 150 chunks để vượt max_corpus_size=100):")
    demo_docs = [f"Document {i}: Đây là tài liệu demo số {i}" for i in range(30)]
    bm25_pipeline.add_documents(demo_docs)
    
    stats_after = bm25_pipeline.get_corpus_stats()
    print(f"\n  Kích thước sau demo: {stats_after['total_chunks']} chunks (max: {stats_after['max_corpus_size']})")
    
    # Thay đổi max_corpus_size
    print("\n🔧 Thay đổi max_corpus_size thành 50:")
    bm25_pipeline.set_max_corpus_size(50)
    
    stats_final = bm25_pipeline.get_corpus_stats()
    print(f"  Kích thước cuối: {stats_final['total_chunks']} chunks")
    
    # Test tìm kiếm
    print("\n🔍 Test tìm kiếm:")
    query = "Thành phố nào là thủ đô Việt Nam?"
    results = bm25_pipeline.retrieve(query, top_k=3, normalize=True)
    
    print(f"\nQuery: '{query}'")
    print("Kết quả (điểm đã chuẩn hóa 0-1):")
    for result in results:
        print(f"  #{result['rank']}: {result['text'][:50]}... (score: {result['score']:.3f})")
    
    # Test batch retrieve
    print("\n📝 Test batch retrieve:")
    queries = [
        "Thành phố ven biển",
        "Di sản thế giới",
        "Dân số Sài Gòn"
    ]
    batch_results = bm25_pipeline.batch_retrieve(queries, top_k=2)
    for q, res in batch_results.items():
        print(f"\n  Query: '{q}'")
        for r in res:
            print(f"    - score: {r['score']:.3f} | {r['text'][:40]}...")
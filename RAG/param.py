class Param:
    """Lớp chứa các hằng số cấu hình cho RAG system."""
    
    # Qdrant configuration
    QDRANT_HOST = "http://localhost:6333/"
    LTM_COLLECTION_NAME = "Long Term Memory"
    VECTOR_SIZE = 384
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # STM configuration
    STM_CONTEXT_WINDOW_SIZE = 10
    STM_COLLECTION_NAME = "Short Term Memory"
    
    # Other configurations
    DEFAULT_SIMILARITY_THRESHOLD = 0.6
    DELETE_SIMILARITY_THRESHOLD = 0.7
    DUPLICATE_SIMILARITY_THRESHOLD = 0.85

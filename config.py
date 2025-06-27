# config.py
from pathlib import Path

class Config:
    # 文件处理配置
    MARKDOWN_DIR = Path("./data")
    CHUNK_MIN_SIZE = 200
    CHUNK_MAX_SIZE = 1500
    
    # Ollama配置
    # OLLAMA_BASE_URL = "http://localhost:9000"
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "bge-m3"
    # EMBEDDING_MODEL = "nomic-embed-text"
    # EMBEDDING_MODEL = "all-minilm"
    # OLLAMA_LLM = "deepseek-r1:7b-qwen-distill-q4_K_M"
    OLLAMA_LLM = "deepseek-r1:1.5b"
    BATCH_SIZE = 5
    MAX_WORKERS = 4
    
    # ChromaDB配置
    CHROMA_DIR = Path("data/chroma_db") / EMBEDDING_MODEL
    COLLECTION_NAME = "technical_docs"
    
    # 缓存配置
    CACHE_DIR = Path("data/cache") 
    EMBEDDING_CACHE = CACHE_DIR / EMBEDDING_MODEL / "embeddings_cache"

config = Config()
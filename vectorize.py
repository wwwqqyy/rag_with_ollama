# vectorize.py
import requests
import chromadb
from typing import List, Dict, Optional
from pathlib import Path
import diskcache
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import uuid
from datetime import datetime
from config import config

class OllamaVectorizer:
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.EMBEDDING_MODEL
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # 初始化缓存
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(config.EMBEDDING_CACHE))
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{self.model}_{text_hash}"
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """向量化单个文本（带缓存）"""
        cache_key = self._get_cache_key(text)
        
        # 检查缓存
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            payload = {
                "model": self.model,
                "prompt": text,
                "options": {"embedding_only": True}
            }
            resp = self.session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            embedding = resp.json().get("embedding")
            
            # 缓存结果
            if embedding:
                self.cache.set(cache_key, embedding)
            return embedding
        except Exception as e:
            print(f"文本向量化失败: {str(e)}")
            return None
    
    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文本（多线程）"""
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = {executor.submit(self.embed_text, text): i 
                      for i, text in enumerate(texts)}
            
            with tqdm(total=len(texts), desc="向量化进度") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result() or []
                    except Exception as e:
                        print(f"处理失败: {str(e)}")
                        results[idx] = []
                    pbar.update(1)
        
        return results

class ChromaDBStore:
    def __init__(self):
        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
    
    def _generate_document_id(self, file_name: str, chunk_idx: int) -> str:
        """生成文档ID"""
        return f"{Path(file_name).stem}_{chunk_idx}"
    # def _generate_document_id(self, file_name: str, chunk_idx: int, chunk_content: str) -> str:
    #     """确定性ID生成（相同输入永远返回相同ID）"""
    #     content_hash = hashlib.md5(chunk_content.encode()).hexdigest()[:8]
    #     return f"{Path(file_name).stem}_{chunk_idx}_{content_hash}"
    
    def add_documents(self, file_name: str, chunks: List[Dict], embeddings: List[List[float]]):
        """添加文档到ChromaDB"""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks和embeddings长度不一致")
        
        # 准备数据
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not embedding:  # 跳过无效的嵌入
                continue
                
            ids.append(self._generate_document_id(file_name, idx))
            documents.append(chunk['content'])
            metadatas.append({
                "source": file_name,
                "heading": chunk.get('heading', ''),
                "chunk_index": idx,
                "processed_at": datetime.now().isoformat()
            })
            embeddings_list.append(embedding)
        
        # 批量添加
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            print(f"已存储 {len(ids)} 个块到ChromaDB")
    
    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """查询相似文档"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        formatted = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            formatted.append({
                "content": doc,
                "metadata": meta,
                "similarity": 1 - dist  # 转换为相似度分数
            })
        
        return formatted
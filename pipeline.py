from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from ollama_api import OllamaAPI
from chunk_md import MarkdownChunker
from vectorize import OllamaVectorizer, ChromaDBStore
from config import config


class RAGPipeline:
    def __init__(self, model_name: str = None):
        """初始化RAG管道
        
        Args:
            model_name: 使用的LLM模型名称
        """
        self.chunker = MarkdownChunker(
            min_chunk_size=config.CHUNK_MIN_SIZE,
            max_chunk_size=config.CHUNK_MAX_SIZE
        )
        self.vectorizer = OllamaVectorizer()
        self.vector_store = ChromaDBStore()
        self.llm = OllamaAPI()
        self.llm_model = config.OLLAMA_LLM

    def process_directory(self):
        """处理整个目录的Markdown文件"""
        if not config.MARKDOWN_DIR.exists():
            raise FileNotFoundError(f"Markdown目录不存在: {config.MARKDOWN_DIR}")
        
        chunks_by_file = self.chunker.chunk_markdown_files(str(config.MARKDOWN_DIR))
        print(f"发现 {len(chunks_by_file)} 个文件在目录: {config.MARKDOWN_DIR}")
        
        for file_name, chunks in chunks_by_file.items():
            print(f"\n处理文件: {file_name} (共{len(chunks)}个块)")
            
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.vectorizer.batch_embed(texts)
            self.vector_store.add_documents(file_name, chunks, embeddings)
        
        print("\n处理完成! 向量已存储到ChromaDB:", config.CHROMA_DIR)
    
    def query(self, question: str, top_k: int = 3) -> List[Dict]:
        """查询向量数据库
        
        Args:
            question: 查询问题
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        query_embedding = self.vectorizer.embed_text(question)
        if not query_embedding:
            return []
        return self.vector_store.query(query_embedding, top_k)
    
    def _build_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """构建完整的生成提示
        
        Args:
            question: 用户问题
            retrieved_docs: 检索到的文档列表
            
        Returns:
            构建好的提示文本
        """
        context = "\n\n".join([
            f"文档 {i+1} (相似度: {doc['similarity']:.2f}):\n"
            f"来源: {doc['metadata']['source']}\n"
            f"标题: {doc['metadata']['heading']}\n"
            f"内容: {doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        return (
            "你是一个专业的油气井控助手，请根据以下检索到的文档内容回答问题。\n"
            "如果文档内容不足以回答问题，请如实告知。回答时请保持专业、准确。\n\n"
            "检索到的文档:\n"
            f"{context}\n\n"
            "用户问题:\n"
            f"{question}\n\n"
            "请用中文给出专业、准确的回答:"
        )

    def generate_response(
            self,
            question: str,
            top_k: int = 3,
            stream: bool = False,
            callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """生成回答(使用generate方法)
        
        Args:
            question: 用户问题
            top_k: 检索结果数量
            stream: 是否使用流式响应
            callback: 流式响应回调函数
            
        Returns:
            包含响应和参考文档的字典
        """
        retrieved_docs = self.query(question, top_k)
        if not retrieved_docs:
            return {
                "response": "抱歉，没有找到相关信息来回答这个问题。",
                "references": []
            }
        
        # 构建完整的生成提示
        prompt = self._build_prompt(question, retrieved_docs)
        
        # 调用generate方法
        response = self.llm.generate(
            model=self.llm_model,
            prompt=prompt,
            stream=stream,
            callback=callback
        )
        
        # 处理响应
        if isinstance(response, dict):
            if 'error' in response:
                return {
                    "response": f"请求失败: {response['error']}",
                    "references": retrieved_docs
                }
            message_content = response.get('response', '')
        else:
            message_content = str(response)
        
        return {
            "response": message_content if message_content else "请求成功但未返回有效内容",
            "references": retrieved_docs
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    
    # 处理文档 只需要第一次运行时候调用
    pipeline.process_directory()
    
    # 预定义第一个问题
    first_question = "石油天然气开采单位应如何编制应急预案？"
    questions_asked = 0
    
    # 交互式问答
    while True:
        if questions_asked == 0:
            # 自动提出第一个问题
            question = first_question
            print(f"\n默认问题: {question}")
        else:
            question = input("\n请输入您的问题(输入q退出): ").strip()
            if question.lower() == 'q':
                break
        
        # 流式输出回调函数
        def stream_callback(chunk: str):
            print(chunk, end="", flush=True)
        
        print("\nAI回答:")
        result = pipeline.generate_response(
            question,
            stream=True,  # 关闭流式以简化调试
            callback=stream_callback
        )
        
        # 打印完整结果
        print("\n完整响应:")
        print(result["response"])
        
        # 显示参考文档
        print("\n参考文档:")
        for i, doc in enumerate(result["references"], 1):
            print(f"\n文档 {i} (相似度: {doc['similarity']:.2f}):")
            print(f"来源: {doc['metadata']['source']}")
            print(f"标题: {doc['metadata']['heading']}")
            print(f"内容: {doc['content'][:200]}...")
        
        questions_asked += 1
        break
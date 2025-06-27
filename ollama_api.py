import requests
import json
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from config import config  # 导入配置


class OllamaAPI:
    def __init__(self, base_url: str = None):
        """初始化Ollama API客户端
        
        Args:
            base_url: Ollama服务地址，如果为None则从config中获取
        """
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(
            self,
            model: str,
            prompt: str,
            stream: bool = False,
            callback: Optional[Callable[[str], None]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """文本生成方法
        
        Args:
            model: 模型名称
            prompt: 提示文本
            stream: 是否使用流式响应
            callback: 流式响应时的回调函数
            **kwargs: 其他参数
            
        Returns:
            生成结果字典
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": kwargs.get("options", {})
        }

        try:
            if stream:
                return self._handle_stream(payload, callback, kwargs.get("timeout", 60))
            else:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=kwargs.get("timeout", 30)
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {str(e)}")
            return {"error": str(e)}

    def _handle_stream(self, payload: Dict, callback: Optional[Callable], timeout: int) -> Dict[str, Any]:
        """处理流式响应
        
        Args:
            payload: 请求负载
            callback: 回调函数
            timeout: 超时时间
            
        Returns:
            流式响应结果
        """
        full_response = ""
        try:
            with self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=timeout
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        if "response" in chunk:
                            if callback:
                                callback(chunk["response"])
                            full_response += chunk["response"]

            return {"response": full_response}
        except Exception as e:
            print(f"流式处理失败: {str(e)}")
            return {"error": str(e)}

    def chat(
            self,
            model: str,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> Dict[str, Any]:
        """对话接口
        
        Args:
            model: 模型名称
            messages: 消息历史
            **kwargs: 其他参数
            
        Returns:
            对话响应结果
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
            "options": kwargs.get("options", {})
        }

        endpoint = "/api/chat"
        try:
            if payload["stream"]:
                return self._handle_stream(payload, kwargs.get("callback"), kwargs.get("timeout", 60))
            else:
                response = self.session.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=kwargs.get("timeout", 30)
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"对话请求失败: {str(e)}")
            return {"error": str(e)}

    def batch_generate(
            self,
            model: str,
            prompts: List[str],
            max_workers: int = None,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """批量生成文本
        
        Args:
            model: 模型名称
            prompts: 提示文本列表
            max_workers: 最大线程数，如果为None则从config中获取
            **kwargs: 其他参数
            
        Returns:
            生成结果列表
        """
        workers = max_workers or config.MAX_WORKERS
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self.generate,
                    model=model,
                    prompt=prompt,
                    **kwargs
                )
                for prompt in prompts
            ]
            return [future.result() for future in futures]


# 使用示例
if __name__ == "__main__":
    # 现在会自动从config.py中获取配置
    ollama = OllamaAPI()
    
    # 对话模式测试
    print("\n=== 对话测试 ===")
    chat_history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好！我是AI助手。"}
    ]
    reply = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=chat_history + [
            {"role": "user", "content": "请介绍你的能力"}
        ]
    )
    print("AI回复:", reply.get("message", {}).get("content", "对话失败"))
    
    # Test the generate method with a simple prompt
    print("\n=== Generate Test ===")
    test_prompt = "请用中文解释一下人工智能的基本概念"
    generation_result = ollama.generate(
        model="deepseek-r1:1.5b",
        prompt=test_prompt
    )
    print("生成结果:")
    print(generation_result.get("response", "生成失败"))
    
    # Test streaming with a callback
    print("\n=== Streaming Generate Test ===")
    def streaming_callback(chunk):
        print(chunk, end='', flush=True)
    
    stream_result = ollama.generate(
        model="deepseek-r1:1.5b",
        prompt="写一篇关于机器学习的200字短文",
        stream=True,
        callback=streaming_callback
    )
    print("\n完整响应:")
    print(stream_result)
    
    # Test batch generation
    print("\n=== Batch Generate Test ===")
    prompts = [
        "用一句话解释神经网络",
        "用一句话解释深度学习",
        "用一句话解释大数据"
    ]
    batch_results = ollama.batch_generate(
        model="deepseek-r1:1.5b",
        prompts=prompts
    )
    for i, result in enumerate(batch_results):
        print(f"\nPrompt {i+1} 结果:")
        print(result.get("response", "生成失败"))
import json
from typing import List, Dict, Set
import numpy as np
from pathlib import Path
from vectorize import ChromaDBStore, OllamaVectorizer
from tqdm import tqdm

      
class RecallEvaluator:
    def __init__(self, chroma_store: ChromaDBStore):
        self.store = chroma_store
        self.vectorizer = OllamaVectorizer()
    
    def _normalize_source(self, path: str) -> str:
        """标准化路径（去掉documents/前缀和目录结构）"""
        path_obj = Path(path)
        # 去掉documents/前缀（如果有）
        if path.startswith("documents/"):
            path = str(Path(*Path(path).parts[1:]))
        # 只保留文件名
        return path_obj.name
    
    def load_test_data(self, test_file: str) -> List[Dict]:
        """加载测试数据（自动处理编码和转义）"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # 尝试修复常见转义问题
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 处理未转义的反斜杠
                content = content.replace('\\', '\\\\')
                content = content.replace('\\\\\"', '\\\"')
                return json.loads(content)
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(test_file, 'r', encoding='gbk') as f:
                return json.load(f)
    
    def compute_recall(self, test_data: List[Dict], top_k: int = 5) -> Dict[str, float]:
        """
        计算召回率（统一使用纯文件名比较）
        
        Args:
            test_data: 已标准化的测试数据
            top_k: 考虑的前K个结果
        """
        recall_scores = []
        coverage = set()
        total_source_docs = set()
        print(len(test_data))
        for item in tqdm(test_data, desc="评估进度"):
            # print(item)
            query = item["question"]
            target_doc = self._normalize_source(item["source_doc"])
            
            # 获取查询向量
            query_embedding = self.vectorizer.embed_text(query)
            if not query_embedding:
                continue
                
            # 执行查询
            results = self.store.query(query_embedding, top_k)
            
            # 标准化检索结果中的source
            retrieved_docs = {
                self._normalize_source(res['metadata']['source']) 
                for res in results
            }
            
            # 计算召回率
            is_relevant = 1 if target_doc in retrieved_docs else 0
            recall_scores.append(is_relevant)
            
            # 统计覆盖率
            total_source_docs.add(target_doc)
            if is_relevant:
                coverage.add(target_doc)
        
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        return {
            "recall@k": avg_recall,
            "coverage": len(coverage) / len(total_source_docs) if total_source_docs else 0,
            "total_queries": len(test_data),
            "relevant_found": sum(recall_scores),
            "details": [
                {
                    "question": item["question"],
                    "expected_doc": self._normalize_source(item["source_doc"]),
                    "retrieved_docs": [
                        self._normalize_source(res['metadata']['source']) 
                        for res in self.store.query(
                            self.vectorizer.embed_text(item["question"]), 
                            top_k
                        )
                    ] if "question" in item else []
                }
                for item in test_data
            ]
        }

def main():
    # 初始化
    store = ChromaDBStore()
    evaluator = RecallEvaluator(store)
    
    # 加载测试数据（自动标准化路径）
    # test_data = [
    #     {
    #         "context": [
    #             "4.4 资质要求\n4. 4. 1 人员\n人员资质要求：\na）检验人员应取得“井控检验员资格证”，且在有效期内；\nb）无损检测人员应取得市场监督管理局或协会颁发的“无损检测人员资格证”，且在有效期内。\n4.4.2 机构\n第三方检验机构应具备独立法人资格并取得省部级及以上检验检测机构授权，授权范围覆盖防喷器、防喷器控制装置、井控管汇。",
    #             "7.2 检验报告\n检验报告要求：\na）定期检验工作完成后，检验人员按照检验实际情况和检验结果，按照本文件规定评定井控装备的安全状况等级，出具检验报告，并且明确下次定期检验的日期；\nb）定期检验报告的编制、审核应符合相关法规或标准的规定。检验报告应在井控装备检测后30个工作日内送交委托方；\nc）定期检验报告的保存期应符合相关法规标准的要求，第三方检验机构报告保存期不少于6年且不低于检验有效期，管理部门应保存报告直到报废为止。"
    #         ],
    #         "question": "井控装备定期检验对检验人员、机构和报告有哪些要求？\n",
    #         "answer": "1. 检验人员需持有有效的“井控检验员资格证”，无损检测人员需持有有效的“无损检测人员资格证”。\n2. 第三方检验机构需具备独立法人资格和省部级及以上检验检测机构授权，授权范围覆盖相关装备。\n3. 检验报告应在检测后30个工作日内送交委托方，保存期不少于6年且不低于检验有效期，管理部门应保存至装备报废。\n\n【总结】检验需专业资质人员和机构完成，报告需及时、规范、长期保存。",
    #         "source_doc": "documents/QSY 02037-2023《井控装备定期检验及分级评定规范》.md"
    #     },
    # ]
    # 或从文件加载：
    test_data = evaluator.load_test_data("data/mergedVersionV2_new.json")
    
    # 计算召回率
    results = evaluator.compute_recall(test_data, top_k=3)
    
    # 打印结果
    print(f"\n评估结果（基于文件名匹配）:")
    print(f"召回率@3: {results['recall@k']:.2%}")
    print(f"文档覆盖率: {results['coverage']:.2%}")
    print(f"匹配详情（显示纯文件名）:")
    # for detail in results["details"]:
    #     print(f"\n问题: {detail['question']}")
    #     print(f"期望文档: {detail['expected_doc']}")
    #     print("检索到的文档:")
    #     for doc in detail['retrieved_docs']:
    #         print(f"- {doc}")

if __name__ == "__main__":
    main()
import re
from typing import List, Dict
from pathlib import Path


class MarkdownChunker:
    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 1500):
        """
        初始化Markdown分块器

        参数:
            min_chunk_size: 最小分块大小(字符数)，小于此值的块会被合并
            max_chunk_size: 最大分块大小(字符数)，超过此值的块会被拆分
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        # 匹配Markdown标题的正则表达式 (1-6级标题)
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        # 匹配Markdown表格的正则表达式
        self.table_pattern = re.compile(r'^\|.+\|$', re.MULTILINE)

    def _is_table(self, text: str) -> bool:
        """检查文本是否是表格"""
        return bool(self.table_pattern.search(text))

    def _split_large_chunk(self, chunk: str) -> List[str]:
        """拆分过大的文本块"""
        if len(chunk) <= self.max_chunk_size or self._is_table(chunk):
            return [chunk]

        # 尝试在段落边界拆分
        paragraphs = re.split(r'\n\s*\n', chunk)
        if len(paragraphs) > 1:
            result = []
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) > self.max_chunk_size and current_chunk:
                    result.append(current_chunk.strip())
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            if current_chunk:
                result.append(current_chunk.strip())
            return result

        # 如果没有段落，则在句子边界拆分
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        if len(sentences) > 1:
            result = []
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) > self.max_chunk_size and current_chunk:
                    result.append(current_chunk.strip())
                    current_chunk = sent
                else:
                    if current_chunk:
                        current_chunk += " " + sent
                    else:
                        current_chunk = sent
            if current_chunk:
                result.append(current_chunk.strip())
            return result

        # 最后手段，按固定长度拆分
        return [chunk[i:i + self.max_chunk_size] for i in range(0, len(chunk), self.max_chunk_size)]

    def chunk_markdown_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        分块Markdown文件

        参数:
            file_path: Markdown文件路径

        返回:
            包含分块信息的字典列表，每个字典包含:
            - 'heading': 标题层级路径 (如 "## 标题1 > ### 标题2")
            - 'content': 分块内容
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 找到所有标题位置
        headings = []
        for match in self.heading_pattern.finditer(content):
            level = len(match.group(1))  # 标题级别 (1-6)
            text = match.group(2).strip()  # 标题文本
            start_pos = match.start()  # 标题开始位置
            headings.append((level, text, start_pos))

        # 如果没有标题，整个文件作为一个块
        if not headings:
            chunks = self._split_large_chunk(content)
            return [{'heading': '', 'content': chunk} for chunk in chunks]

        # 根据标题位置分割内容
        chunks = []
        prev_pos = 0
        heading_stack = []  # 用于跟踪标题层级

        for i, (level, text, start_pos) in enumerate(headings):
            # 获取当前标题之前的内容
            chunk_content = content[prev_pos:start_pos].strip()
            if chunk_content:
                # 处理前一个块
                chunk_heading = " > ".join([h[1] for h in heading_stack]) if heading_stack else ""
                chunks.append({
                    'heading': chunk_heading,
                    'content': chunk_content
                })

            # 更新标题栈
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))

            prev_pos = start_pos

        # 添加最后一个标题之后的内容
        last_chunk_content = content[prev_pos:].strip()
        if last_chunk_content:
            chunk_heading = " > ".join([h[1] for h in heading_stack]) if heading_stack else ""
            chunks.append({
                'heading': chunk_heading,
                'content': last_chunk_content
            })

        # 合并小分块和拆分大分块
        processed_chunks = []
        current_chunk = ""
        current_heading = ""

        for chunk in chunks:
            # 如果当前块为空，初始化
            if not current_chunk:
                current_chunk = chunk['content']
                current_heading = chunk['heading']
                continue

            # 如果标题相同且合并后不超过最大大小，则合并
            if (chunk['heading'] == current_heading and
                    len(current_chunk) + len(chunk['content']) < self.max_chunk_size and
                    not self._is_table(current_chunk) and
                    not self._is_table(chunk['content'])):
                current_chunk += "\n\n" + chunk['content']
            else:
                # 处理当前积累的块
                split_chunks = self._split_large_chunk(current_chunk)
                for split_chunk in split_chunks:
                    processed_chunks.append({
                        'heading': current_heading,
                        'content': split_chunk
                    })
                # 开始新块
                current_chunk = chunk['content']
                current_heading = chunk['heading']

        # 处理最后一个积累的块
        if current_chunk:
            split_chunks = self._split_large_chunk(current_chunk)
            for split_chunk in split_chunks:
                processed_chunks.append({
                    'heading': current_heading,
                    'content': split_chunk
                })

        # 过滤空块
        processed_chunks = [chunk for chunk in processed_chunks if chunk['content'].strip()]

        return processed_chunks

    def chunk_markdown_files(self, dir_path: str) -> Dict[str, List[Dict[str, str]]]:
        """
        分块目录中的所有Markdown文件

        参数:
            dir_path: 包含Markdown文件的目录路径

        返回:
            字典，键为文件名，值为该文件的分块列表
        """
        md_files = Path(dir_path).glob('*.md')
        result = {}

        for md_file in md_files:
            # print(str(md_file))
            chunks = self.chunk_markdown_file(str(md_file))
            result[md_file.name] = chunks

        return result


# 使用示例
if __name__ == "__main__":
    chunker = MarkdownChunker(min_chunk_size=200, max_chunk_size=1500)

    # 处理单个文件
    chunks = chunker.chunk_markdown_file("data/QSY 02552-2022 钻井井控技术规范.md")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"Heading: {chunk['heading']}")
        print(f"Content: {chunk['content'][:200]}...")  # 只打印前200字符
        print("-" * 50)

    # 处理整个目录
    # all_chunks = chunker.chunk_markdown_files("markdown_docs")
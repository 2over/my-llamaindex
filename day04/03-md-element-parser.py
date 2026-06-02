from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.readers.file import FlatReader
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()
model = "qwen-plus"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

Settings.llm = DashScope(model=model, api_key=api_key, api_base_url=api_base_url)
Settings.embed_model =  HuggingFaceEmbedding(model_name=r"BAAI/bge-small-zh-v1.5")


# 自定义你的提示词
# 建议: 明确告诉AI保持简介，并提取关键的关键词(如命名、字段名)
MY_CUSTOM_SUMMARY_QUERY = (
    "你是一个技术文档解析助手。请提取以下Markdown表格或内容的极简摘要"
    "要求： 1.严禁啰嗦; 2.必须包含表格中的关键实体词(如API路径、参数名、状态码);"
    "3.如果是代码相关内容, 请保留具体的命令名称。请用中文摘要"
)

# 读取文件 + 解析文档
md_docs = FlatReader().load_data(Path("../data/test.md"))
parser = MarkdownElementNodeParser(include_prev_next_rel=True, summary_query_str=MY_CUSTOM_SUMMARY_QUERY)
nodes = parser.get_nodes_from_documents(md_docs)
print(nodes)

# 构建向量索引
index = VectorStoreIndex(nodes)

retriever = index.as_retriever(similarity_top_k=5)

# retriever.retrieve("张三多少岁")
# 创建查询引擎
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# 3.组合成查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

# 测试查询
print("\n-- 测试查询: 1: 针对表格数据 ---")
response = query_engine.query("张三多少岁?")

print(response)

print("\n-- 测试查询: 1: 针对文本/代码内容 ---")
response = query_engine.query("李四在哪个城市?")
print(response)

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 初始化分割器
splitter = SentenceSplitter(
    chunk_size=512, # 分割长度
    chunk_overlap=50, # 重叠长度
    paragraph_separator="\r\n\r\n", # 段落分隔符 第一优先级
    secondary_chunking_regex="[^，。；！？]+[，。；！？]?" # 二级切片正则表达式 第二优先级
)

# 读取文件
documents = SimpleDirectoryReader(input_files=['../data/西游记第一回.txt']).load_data()


# 分割文段
nodes = splitter.get_nodes_from_documents(documents)
for node in nodes:
    print(node.text, "----"*10)
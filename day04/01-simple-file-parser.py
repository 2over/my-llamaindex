from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path


# 将文件从本地读取到应用中(只是将文件中的原始文本提取)
documents = FlatReader().load_data(Path("../data/西游记第一回.txt"))
parser = SimpleFileNodeParser()
# 将文件解析成节点
nodes = parser.get_nodes_from_documents(documents)
# SimpleFileNodeParser只会将文档解析成一个节点
print(nodes)
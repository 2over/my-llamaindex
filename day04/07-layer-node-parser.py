from llama_index.core.base.embeddings.base import similarity
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 读取数据
documents = SimpleDirectoryReader(input_files=['../data/西游记第一回.txt']).load_data()

# 进行层次节点解析器 chunk_sizes = 每层目标Tokens(从粗到细)
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 300],
    chunk_overlap=70
)

# 文档转换成节点
nodes = node_parser.get_nodes_from_documents(documents)

# 会根据子节点合并父节点
# Settings.embed_model =  HuggingFaceEmbedding(model_name=r"BAAI/bge-small-zh-v1.5")

Settings.embed_model = HuggingFaceEmbedding(model_name=r"BAAI/bge-small-zh-v1.5")

# 获取叶节点(最细粒度)没有子节点:
from llama_index.core.node_parser import get_leaf_nodes
leaf_nodes = get_leaf_nodes(nodes) # 获取到没有子节点的节点

# 构建存储上下文: 包括所有节点
from llama_index.core.storage.docstore import SimpleDocumentStore

# 创建文档存储
docstore = SimpleDocumentStore()

# 添加文档
docstore.add_documents(nodes)
# 创建需要存储的上下文
storage_context = StorageContext.from_defaults(docstore=docstore)

# 构建基础向量检索索引: 仅对叶节点构建
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
)

base_retriever = base_index.as_retriever(similarity_top_k=6)

# 构建AutoMergingRetriever

retriever = AutoMergingRetriever(
    vector_retriever=base_retriever,
    storage_context=storage_context,
    simple_ratio_thresh=0.5, # 控制合并阈值
    verbose=True, # 显示合并日志
)

# 查询
query_str = "孙悟空的师父是谁？"
nodes_returned = retriever.retrieve(query_str)
print(f"Retrieved {len(nodes_returned)} nodes")

for node in nodes_returned:
    print("------")
    print(node.get_content())

base_nodes_returned = base_retriever.retrieve(query_str)
print(f"Retrieved  {len(base_nodes_returned)} nodes")

for node in base_nodes_returned:
    print("------")
    print(node.get_content())
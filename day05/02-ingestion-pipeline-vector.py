from llama_index.core.base.embeddings.base import similarity
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
import chromadb

from load_llm_models import get_llm

llm, embed_model = get_llm()

# 定义数据连接器去读数据
documents = SimpleDirectoryReader(input_files=["../data/西游记第一回.txt"]).load_data()
# 定义内存的chromadb
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.get_or_create_collection("quickstart")
#创建Chroma向量数据库对象
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


# 定义文本分割器
text_splitter = SentenceSplitter(chunk_size=500, chunk_overlap=30)

# 创建数据摄入管道
pipeline = IngestionPipeline(
    transformations=[text_splitter, embed_model],
    vector_store=vector_store
)

# 执行管道
nodes = pipeline.run(documents=documents)


# 创建索引对象
index = VectorStoreIndex.from_vector_store(vector_store)

# 创建检索器
retriever = index.as_retriever(similarity_top_k=3)
print(retriever.retrieve("菩提祖师住在哪里？"))


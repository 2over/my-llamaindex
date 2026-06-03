from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
import chromadb
import time
from load_llm_models import get_llm

llm, embed_model = get_llm()

# 定义数据连接器去读取数据
documents = SimpleDirectoryReader(input_files=["../data/西游记第一回.txt"]).load_data()
# 定义本地化的向量化
chroma_client = chromadb.PersistentClient("./chroma")
chroma_collection = chroma_client.get_or_create_collection("quickstart")
# 创建Chroma向量数据库对象
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 定义文本分割器
text_splitter = SentenceSplitter(chunk_size=500, chunk_overlap=30)

# 创建数据摄入管道
pipeline = IngestionPipeline(
    transformations=[text_splitter, embed_model],
    vector_store=vector_store,
)

# 开始事件
start = time.time()

# 执行管道

pipeline.run(documents=documents)
# 统计文档加载时间
time2 = time.time() - start
print(f">>>>>>第一次处理文档, 耗时: {time2: .2f}秒")

# 将这个管道持久化到本地
pipeline.persist("./pipeline_storage")

# 加载和恢复状态
new_pipeline = IngestionPipeline(
    transformations=[text_splitter, embed_model],
    vector_store=vector_store
)

# 从缓存中读取持久化管道数据
new_pipeline.load("./pipeline_storage")
# 开始时间
new_start = time.time()

# 优于缓存的存在会立即执行
nodes = new_pipeline.run(documents=documents)

# 统计文档加载的时间
new_time2 = time.time() - new_start
print(f">>>> 缓存命中，跳过了重复处理，耗时:{new_time2: .2f}秒")

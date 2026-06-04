import os
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import SentenceSplitter


from load_llm_models import get_llm

llm, embed_model = get_llm()

# 配置常量
STORAGE_DIR = "./pipeline_doc_storage"
DATA_DIR = "..//data1"


# ---- 核心逻辑: 封装成一个增量运行函数 ----
def run_incremental_ingestion(data_path, storage_path):
    # 1. 准备基础组件
    # 如果存储目录存在，则加载旧的docstore, 否则新建
    if os.path.exists(storage_path):
        print("--- 发现现有存储， 正在加载增量状态 ---")
        docstore = SimpleDocumentStore.from_persist_dir(storage_path)
    else:
        print("---- 未发现存储，初始化全新管道 ----")
        docstore = SimpleDocumentStore()


    # 2. 构造管道
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size = 512, chunk_overlap=20),
            embed_model,
        ],
        docstore=docstore,
        # 重点: 设置文档存储策略为 UPSERT(更新或插入)
        # 这回让管道自动对比Hash, 如果文档没变，它就不会出现在返回的nodes中
        docstore_strategy="upserts",
    )

    # 3.加载文档（filename_as_id=True保证了ID的稳定性）
    documents = SimpleDirectoryReader(data_path, filename_as_id=True).load_data()

    # 4.执行管道
    # 注意: 此时nodes 只包含【新增】或【被修改】后重新生成的节点
    nodes = pipeline.run(documents=documents, show_progress=True)

    # 5. 持久化(保存docstore和cache状态)
    pipeline.persist(storage_path)

    return nodes


# --- 演示步骤 ---
print("\n[第一轮运行]")
nodes1 = run_incremental_ingestion(DATA_DIR, STORAGE_DIR)
print(f"实际摄取的新节点数: {len(nodes1)}")

# 模拟: 增加一个文件
with open(f"{DATA_DIR}/t4.txt", 'w', encoding='utf-8') as f:
    f.write("这是测试文件-新增内容")

# 第二轮运行
print("\n[第二轮运行]")
nodes2 = run_incremental_ingestion(DATA_DIR, STORAGE_DIR)
print(f"实际摄取的新节点数: {len(nodes2)}")
# 此时nodes2的长度应该远小于总数，仅包含t4.txt的内容

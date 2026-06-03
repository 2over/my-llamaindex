from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope
from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv

import os
import torch

load_dotenv()

# 检查GPU可用性
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_llm(model: str = "qwen-plus"):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    api_base_url = os.getenv("DASHSCOPE_BASE_URL")


    # LLamaIndex默认使用
    llm = DashScope(model_name=model, api_key=api_key, api_base_url=api_base_url, is_chat_model=True)
    Settings.llm = llm

    # 加载本地的嵌入模型
    embed_model =  HuggingFaceEmbedding(model_name=r"BAAI/bge-small-zh-v1.5")

    # 设置默认的向量模型为本地模型
    Settings.embed_model = embed_model

    return llm, embed_model


def get_deepseek_llm(model: str = "deepseek-chat"):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_base_url = os.getenv("DEEPSEEK_BASE_URL")

    # LlamaIndex默认使用的大模型被替换为百炼
    embed_model = HuggingFaceEmbedding(model_name=r"BAAI/bge-small-zh-v1.5", device=device, embed_batch_size=2)


    # 设置默认的向量模型为本地模型
    Settings.embed_model = embed_model

    return llm, embed_model


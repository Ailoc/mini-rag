from collections import defaultdict
import os

# 这三行必须放在最上面，加载任何 huggingface 东西之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 关键！国内加速镜站
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 超时改成 5 分钟
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/huggingface_cache")
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
model = ChatOpenAI(model="doubao-seed-1-6-lite-251015")
presist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
    persist_directory=presist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

class QueryVariations(BaseModel):
    queries: List[str]

original_query = "活着小说中富贵的儿子有庆是什么原因死的？"
llm_with_tools = model.with_structured_output(QueryVariations, method="function_calling")

prompt = f"""请你生成3个不同的查询变体，用于更好的在文档中查询出相关的信息。
原始查询如下：
{original_query}
请你返回三个从不同的角度阐述原始查询的新的查询
"""
response = llm_with_tools.invoke(prompt)

assert type(response) == QueryVariations
variation_queries = response.queries

print("生成的查询变体为：")
for query in variation_queries:
    print(query)

retrieval = db.as_retriever(search_kwargs={"k": 5})
all_docs = []
for i, query in enumerate(variation_queries, 1):
    docs = retrieval.invoke(query)
    all_docs.append(docs)
print("="*30)

def reciprocal_rank_fusion(chunk_lists, k=60):
    rrf_score = defaultdict(float)
    all_unique_docs = {}

    chunk_id_map = {}
    chunk_counter = 1
    for i, chunks in enumerate(chunk_lists, 1):
        for position, chunk in enumerate(chunks, 1):
            chunk_content = chunk.page_content
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            all_unique_docs[chunk_content] = chunk
            position_score = 1 / (k + position)
            rrf_score[chunk_content] += position_score

    sorted_chunks = sorted(
        [(all_unique_docs[chunk_content], score) for chunk_content, score in rrf_score.items()],
        key=lambda x: x[1], reverse=True
    )

    return [tup[0] for tup in sorted_chunks]

chunks_doc = reciprocal_rank_fusion(all_docs)

combined_input = f'''请结合文档中的内容，回答我的问题。
文档内容如下：
{chr(10).join([f"{doc.page_content}" for doc in chunks_doc])}
我的问题是：{original_query}
'''

messages = [
    SystemMessage(content="你是一个乐于助人的助手！"),
    HumanMessage(content=combined_input),
]
results = model.invoke(messages)
print(results.content)

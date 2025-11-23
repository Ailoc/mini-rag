import os

from dotenv import load_dotenv
from langchain_chroma import Chroma

# 这三行必须放在最上面，加载任何 huggingface 东西之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 关键！国内加速镜站
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 超时改成 5 分钟
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/huggingface_cache")
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

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

query = "富贵的儿子有庆是什么原因死的？"

retrieval = db.as_retriever(search_kwargs={"k": 5})
# retrieval = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 3, "score_threshold": 0.3},
# )
relevant_docs = retrieval.invoke(query)

combined_input = f'''请结合文档中的内容，回答我的问题。
文档内容如下：
{chr(10).join([f"{doc.page_content}" for doc in relevant_docs])}
我的问题是：{query}
'''

model = ChatOpenAI(model="doubao-seed-1-6-lite-251015")

messages = [
    SystemMessage(content="你是一个乐于助人的助手！"),
    HumanMessage(content=combined_input),
]
results = model.invoke(messages)
print(results.content)
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")

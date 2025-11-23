import os

# # 在导入其他库之前先禁用 SOCKS 代理
# _proxy_backup = {}
# for key in [
#     "HTTP_PROXY",
#     "HTTPS_PROXY",
#     "http_proxy",
#     "https_proxy",
#     "ALL_PROXY",
#     "all_proxy",
# ]:
#     if key in os.environ:
#         _proxy_backup[key] = os.environ.pop(key)
#

# 这三行必须放在最上面，加载任何 huggingface 东西之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 关键！国内加速镜站
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 超时改成 5 分钟
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/huggingface_cache")
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from volcenginesdkarkruntime import Ark

load_dotenv()


def load_documents(docs_path="docs"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exists.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "gb18030"},
    )
    documents = loader.load()
    # print(documents)

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files in {docs_path}.")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", "。", ".", "?", "!", "？", "！", "，"],
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    # 创建自定义的 Embedding 类来使用 Ark
    # class ArkEmbeddings(Embeddings):  # 继承 Embeddings 基类
    #     def __init__(self):
    #         self.client = Ark(
    #             api_key=os.environ.get("OPENAI_API_KEY"),
    #             base_url=str(os.environ.get("OPENAI_API_BASE")),
    #         )

    #     def embed_documents(self, texts: list[str]) -> list[list[float]]:
    #         """向量化多个文档"""
    #         batch_size = 256
    #         all_embeddings = []

    #         for i in range(0, len(texts), batch_size):
    #             batch = texts[i : i + batch_size]
    #             response = self.client.embeddings.create(
    #                 model="doubao-embedding-text-240715",
    #                 input=batch,
    #                 encoding_format="float",
    #             )
    #             batch_embeddings = [item.embedding for item in response.data]
    #             all_embeddings.extend(batch_embeddings)

    #         return all_embeddings

    #     def embed_query(self, text: str) -> list[float]:
    #         """向量化单个查询"""
    #         response = self.client.embeddings.create(
    #             model="doubao-embedding-text-240715",
    #             input=[text],
    #             encoding_format="float",
    #         )
    #         return response.data[0].embedding

    # 使用 LangChain 的 Chroma
    # embeddings = ArkEmbeddings()
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return vectorstore


def main():
    documents = load_documents(docs_path="docs")
    chunks = split_documents(documents)
    _ = create_vector_store(chunks)


if __name__ == "__main__":
    main()

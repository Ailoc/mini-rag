import os

from dotenv import load_dotenv
from langchain_chroma import Chroma

# 这三行必须放在最上面，加载任何 huggingface 东西之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 关键！国内加速镜站
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 超时改成 5 分钟
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/huggingface_cache")
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
model = ChatOpenAI(model="doubao-seed-1-6-lite-251015")
history = []
def get_answer(question):
    if history:
        messages = [SystemMessage(
            content=f"""你是一个非常聪明的助手，可以从你的对话历史中总结信息，
            对于用户提出的问题，你总能给出搜索语句用于在文档库中进行相应内容的搜索.
            请你根据用户的输入，给出最终的搜索语句
            """
        )] + history + [HumanMessage(content=f"new question: {question}")]
    else:
        messages = [HumanMessage(content=f"new question: {question}")]

    search_question = model.invoke(messages).content
    retrieval = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retrieval.invoke(str(search_question).strip())

    combined_input = f'''请结合文档中的内容，回答我的问题。
    文档内容如下：
    {chr(10).join([f"{doc.page_content}" for doc in relevant_docs])}
    我的问题是：{question}
    '''

    messages = [
        SystemMessage(content="你是一个乐于助人的助手，可以根据用户给定的相关信息给出最合适的回答")
    ] + history + [HumanMessage(content=combined_input)]

    result = model.invoke(messages).content
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=result))

    return result

def start_chat():
    while True:
        print("请输入你的问题：按quit结束！")
        question = input("User: ")
        if question.lower() == "quit":
            return
        res = get_answer(question)
        print("Assistant:", res)

if __name__ == "__main__":
    start_chat()

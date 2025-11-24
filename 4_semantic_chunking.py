import os

# 这三行必须放在最上面，加载任何 huggingface 东西之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 关键！国内加速镜站
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 超时改成 5 分钟
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/huggingface_cache")
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
load_dotenv()

llm = ChatOpenAI(model="doubao-seed-1-6-lite-251015")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

text = f"""
"做牛耕田，做狗看家，做和尚化缘，做鸡报晓，做女人织布，哪只牛不耕田？这可是自古就有的道理，走呀，走呀。"
　　疲倦的老牛听到老人的吆喝后，仿佛知错般地抬起了头，拉着犁往前走去。
　　我看到老人的脊背和牛背一样黝黑，两个进入垂暮的生命将那块古板的田地耕得哗哗翻动，犹如水面上掀起的波浪。
　　随后，我听到老人粗哑却令人感动的嗓音，他唱起了旧日的歌谣，先是口依呀啦呀唱出长长的引子，接着出现两句歌词--
　　皇帝招我做女婿，路远迢迢我不去。
　　因为路途遥远，不愿去做皇帝的女婿。老人的自鸣得意让我失声而笑。可能是牛放慢了脚步，老人又吆喝起来：
　　"二喜，有庆不要偷懒；家珍，凤霞耕得好；苦根也行啊。"
　　一头牛竟会有这么多名字？我好奇地走到田边，问走近的老人：
　　"这牛有多少名字？"
　　老人扶住犁站下来，他将我上下打量一番后问：
　　"你是城里人吧？"
　　"是的。"我点点头。
　　老人得意起来，"我一眼就看出来了。"
　　我说："这牛究竟有多少名字？"
　　老人回答："这牛叫福贵，就一个名字。"
　　"可你刚才叫了几个名字。"
　　"噢--"老人高兴地笑起来，他神秘地向我招招手，当我凑过去时，他欲说又止，他看到牛正抬着头，就训斥它：
　　"你别偷听，把头低下。"
　　牛果然低下了头，这时老人悄声对我说：
　　"我怕它知道只有自己在耕田，就多叫出几个名字去骗它，它听到还有别的牛也在耕田，就不会不高兴，耕田也就起劲啦。"
　　老人黝黑的脸在阳光里笑得十分生动，脸上的皱纹欢乐地游动着，里面镶满了泥土，就如布满田间的小道。
　　这位老人后来和我一起坐在了那棵茂盛的树下，在那个充满阳光的下午，他向我讲述了自己。
　　四十多年前，我爹常在这里走来走去，他穿着一身黑颜色的绸衣，总是把双手背在身后，他出门时常对我娘说：
　　"我到自己的地上去走走。"
　　我爹走在自己的田产上，干活的佃户见了，都要双手握住锄头恭敬地叫一声：
　　"老爷。"
"""

semantic_splitting = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)

chunks = semantic_splitting.split_text(text)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {len(str(chunk))} chars")
    print(chunk)
    if i == 5:
        break

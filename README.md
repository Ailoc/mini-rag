# RAG是什么？
RAG（Retrieval-Augmented Generation）是一种结合了检索和大语言模型的技术，通过给大语言模型提供额外的信息，例如文档、数据库等，使得大语言模型可以更好地生成回答。

> 为什么需要RAG？
>
> 因为大语言模型的上下文窗口是有限的，无法应对超长的文本信息。

# RAG系统主要分为两个部分
- 知识的构建(Ingestion pipeline) -> 信息摄取管道
    
      source documents -> chunking -> embedding -> vector database
                                          |
                                词语、句子以及图像等的数学表示
- 检索阶段(Retrieval Pipeline)

      Query -> Retrieval -> chunks -> LLM ->Answer
        |                      ^       
        |______________________|

# 文本分块策略
- *CharacterTextSplitter*
可以自定义分割符，速度快
- *RecursiveCharacterTextSplitter*
它会尝试在自然的文本边界处进行分割，例如句子、段落、单词等位置。如果分块太大，它会自然地回退，相比第一种保留了更多的上下文
- *Document-Sepcific Splitting*(尊重文档结构的分割方式)
比如PDF、Markdown、csv等文档
- *Semantic Splitting*(content-aware boundaries)
通过embedding检测语义变化，将语义相近的文本划分到同一个chunk中
- *Agentic Splitting*(AI-powered chunking)
通过大语言模型对内容进行分析，决定最优的分割策略
---
# *CharacterTextSplitter & RecursiveCharacterTextSplitter*
CharacterTextSplitter对文本的划分主要有两个阶段
1. 默认按照（\n\n）分割文本，当然可以自己指定分割符，但是只能指定一个分割符
2. 将第一阶段分割好的文本进行合并，直到达到最大的文本大小

这将导致有时chunk会很大，或者会在一个比较尴尬的位置对文本进行了划分，例如将一个完整的句子分割在了不同的chunk中

RecursiveCharacterTextSplitter可以指定多个分割符，当文本分割的内容超过了chunk设置的值时，会依次选择后续的分割符对文本进行分割，相比第一种方法更加灵活。

# *Semantic Chunker*
语义分块通过找到段落中话题的自然转变位置来分割语句，而不是随机的通过单词计数的方式。Semantic Chunker的大体工作方式分为如下几个阶段。
1. Encode: 将每个句子进行Embedding。
2. Compare: 计算相邻句子之间的相似度得分。
3. Split： 在相似度显著下降的地方创建分割点。

```python
from langchain_experimental.text_splitter import SemanticChunker
semantic_splitting = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)
```
其中的percentile用于确定分割位置，首先计算相邻句子之间的相似度分数，排序后breakpoint_threshold_amount * 句子数量得到具体的分割分数标准，将高于分割标准的连续几个句子划分到一个chunk中。

> 这种分割方式主要的问题是不同的文档语义相似度具有明显的差别，这有可能导致不同的文档划分出的chunk数量显著不同，甚至有可能将整个文档划分到一个chunk中。

# Agentic chunking
利用大语言模型分析文档内容并进行分块，但是这种方式成本高昂，而且会受到上下文窗口的限制。

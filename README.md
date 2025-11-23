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

# *CharacterTextSplitter*

LLM的3种架构:Encoder-only,Decoder-only,Encoder-Decoder  
在Transformer模型中,编码器负责理解和提取输入文本中的相关信息,并且用  
自注意力机制来理解文本中的上下文关系  
编码器的输出是输入文本的连续表示,通常称为嵌入(embedding),这种嵌入包含了  
编码器从文本中提取的所有有用信息,并以一种可以被模型处理的格式表示  
这个嵌入然后被传递给解码器  
解码器的任务是根据从编码器收到的嵌入来生成翻译后的文本  
解码器也使用自注意力机制,以及编码器-解码器注意力机制,来生成翻译的文本  

LLMs中有的只有编码器encoder-only,有的只有解码器decoder-only  
有的是2者混合encoder decoder hybrid  
三者都属于Seq2Seq,sequence to sequence  
并且字面意思虽然只有编码器,实际上LLMs是能decoder一些文本和token的,也算是decoder  
不过encoder-only类型的LLM不像decoder-only和encoder-decoder那些有自回归autoregressive,encoder-only  
集中于理解输入的内容,并做针对特定任务的输出  
自回归指输出的内容是根据已生成的token做上下文理解后一个token一个token输出的  

encoder-only类型的更擅长做分类  
encoder-decoder类型的擅长输出强烈依赖输入的,比如翻译和文本总结  
其他类型的就用decoder-only,如各种Q&A  
虽然encoder-only没有decoder-only类型的流行,但也常用于模型预训练  

encoder-only:如BERT  
encoder-decoder:如T5  
decoder-only:市面上大多数大模型  

Encoder-only架构的LLMs更擅长对文本内容进行分析,分类，包括情感分析,命名实体识别  
Decoder主要是为了预测下一个输出的内容/token是什么,并把之前输出的内容/token作为上下文学习  
实际上,decoder-only模型在分析分类上也和encoder-only的LLM一样有效  

Encoder-Decoder混合  
这种架构的LLM通常充分利用上面两种类型的优势,采用新的技术和架构调整来优化表现  
这种主要用于NLP,即理解输入的内容NLU,又能处理并生成内容NLG,尤其擅长处理输入  
和输出序列之间存在复杂映射关系的任务,以及捕捉两个序列中元素之间关系至关重要的任务  

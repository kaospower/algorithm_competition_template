Transformer模型详解(2024,https://zhuanlan.zhihu.com/p/338817680)  
Transformer由论文Attention is All You Need提出,现为谷歌云TPU推荐的参考模型  
Tensorflow和PyTorch上均有其参考代码  

Transformer由Encoder和Decoder两个部分组成,Encoder和Decoder都包含6个block  
第一步:获取输入句子的每一个单词的表示向量X,X由单词的Embedding和单词位置的Embedding相加得到  
第二步:将得到的单词表示向量矩阵传入Encoder中,经过6个Encoder block后可以得到句子所有单词的编码信息矩阵C,  
单词向量矩阵用$X_{n\times d}$表示,n是句子中单词个数,d是表示向量的维度  
每一个Encoder block输出的矩阵维度与输入完全一致  
第三步:将Encoder输出的编码信息矩阵C传递到Decoder中,Decoder依次根据当前翻译过的单词1~i翻译下一个单词i+1,  
在使用的过程中,翻译到单词i+1的时候需要通过Mask操作遮盖住i+1之后的单词  

细节  
单词Embedding    
可以采用Word2Vec,Glove等算法预训练得到,也可以在Transformer中训练得到  

位置Embedding  
Transformer不采用RNN的结构,使用全局信息,不能利用单词的顺序信息,而这部分信息对于NLP来说非常重要.  
Transformer中使用位置Embedding保存单词在序列中的相对或绝对位置  
位置Embedding用PE(Positional Encoding,位置编码)表示,PE的维度与单词Embedding是一样的.PE可以通过训练得到,也可以使用某种公式计算得到,Transformer中采用了后者  
$PE(pos,2i)=sin(pos/10000^{2i/d})$  
$PE(pos,2i+1)=cos(pos/10000^{2i/d})$  
其中,pos表示单词在句子中的位置,d表示PE的维度,2i表示偶数维度,2i+1表示奇数维度  
使用这种公式计算PE有以下好处:  
使PE能够适应比训练集里面所有句子更长的句子  
可以让模型容易计算出相对位置  
将单词的词Embedding和位置Embedding相加,就可以得到单词的表示向量x,x就是Transformer的输入  

Self-Attention(自注意力)  
Multi-Head Attention(多头注意力),是由多个Self-Attention组成的  
Encoder block包含一个Multi-Head Attention,Decoder block包含两个Multi-Head Attention  
Multi-Head Attention上方还包括一个Add&Norm层  
Add表示残差连接用于防止网络退化,Norm表示Layer Normalization(层归一化),用于对每一层的激活值进行归一化  

Self-Attention结构  
在计算的时候需要用到矩阵Q(查询),K(键值),V(值).在实际中,Self-Attention接受的是输入(单词的表示向量x组成的矩阵x)  
或上一个Encoder block的输出.而Q,K,V正是通过Self-Attention的输入进行线性变换得到的  

Q,K,V的计算  
Self-Attention的输入用矩阵X进行表示,可以使用线性变换矩阵$W_Q,W_K,W_V$计算得到Q,K,V.  
X,Q,K,V的每一行都表示一个单词  

Self-Attention的输出  
得到矩阵Q,K,V之后就可以计算出Self-Attention的输出了.  
计算公式如下:  
$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$  
$d_k是Q,K矩阵的列数,即向量维度$  
公式中计算矩阵Q,K每一行向量的内积,为了防止内积过大,因此除以$d_k$的平方根.  
Q乘以K的转置后,得到的矩阵行列数都为n,n为句子单词数,这个矩阵可以表示单词之间的attention强度.  
得到$QK^T$之后,使用Softmax计算每一个单词对于其他单词的attention系数,公式中的Softmax是对  
矩阵的每一行进行Softmax,即每一行的和都变为1  
得到Softmax矩阵之后可以和V相乘,得到最终的输出Z  
Softmax矩阵的第1行表示单词1与其他所有单词的attention系数,最终单词1的输出$Z_1$等于所有单词i的值$V_i$  
根据attention系数的比例加在一起得到  

Multi-Head Attention  
Multi-Head Attention是由多个Self-Attention组合形成的  
Multi-Head Attention包含多个Self-Attention层,首先将输入X分别传递到h个不同的Self-Attention中,  
计算得到h个输出矩阵Z.  
Multi-Head Attention将它们拼接在一起(Concat),然后传入一个Linear层,得到最终输出Z  
Multi-Head Attention输出的矩阵Z与输入的矩阵X的维度是一样的  

Add&Norm  
Add&Norm层由Add和Norm两部分组成,计算公式如下:  
$LayerNorm(X+MultiHeadAttention(X))$  
$LayerNorm(X+FeedForward(X))$  
其中X表示Multi-Head Attention或者Feed Forward的输入,MultiHeadAttention(X)  
和FeedForward(X)表示输出(输出与输入X维度是一样的)  
Add指X+MultiHeadAttention(X),是一种残差连接,通常用于解决多层网络训练问题,可以让网络  
只关注当前差异的部分,在ResNet中经常用到  
Norm指Layer Normalization,通常用于RNN结构,Layer Normalization会将每一层神经元的输入  
都转成均值方差都一样的,这样可以加快收敛  

Feed Forward  
Feed Forward层比较简单,是一个两层的全连接层,第一层的激活函数为ReLU,第二层不使用激活函数  
对应的公式如下:  
$max(0,XW_1+b_1)W_2+b_2$  
X是输入,Feed Forward最终得到的输出矩阵的维度与X一致  

组成Encoder  
Multi-Head Attention,Feed Forward,Add&Norm就可以构造出一个Encoder block  
Encoder block接受输入矩阵$X_{(n\times d)}$,并输出一个矩阵$O_{(n\times d)}$  
通过多个Encoder block叠加就可以组成Encoder  
第一个Encoder block的输入为句子单词的表示向量矩阵,后续Encoder block的输入是前一个  
Encoder block的输出,最后一个Encoder block输出的矩阵就是编码信息矩阵C,这一矩阵后续会  
用到Decoder中  

Decoder结构  
包含两个Multi-Head Attention层  
第一个Multi-Head Attention层采用了Masked操作  
第二个Multi-Head Attention层的K,V矩阵使用Encoder的编码信息矩阵C进行计算,  
而Q使用上一个Decoder block的输出计算  
最后有一个Softmax层计算下一个翻译单词的概率  

第一个Multi-Head Attention  
Decoder block的第一个Multi-Head Attention采用了Masked操作,通过Masked操作  
可以防止第i个单词得到i+1个单词之后的信息  
Decoder可以在训练的过程中使用Teacher Forcing(教师强制)并且并行化训练  
Mask操作是在Self-Attention的Softmax之前使用的  
第一步:  
是Decoder的输入矩阵和Mask矩阵  
在Mask可以发现单词0只能使用单词0的信息,单词1可以使用单词0,1的信息,即只能使用之前的信息  
第二步:  
接下来的操作和之前的Self-Attention一样,通过输入矩阵X计算得到Q,K,V矩阵.然后计算$Q和K^T的乘积QK^T$  
第三步:  
在得到$QK^T$之后需要进行Softmax,计算attention score,在Softmax之前需要使用Mask矩阵遮挡住每一个单词之后的信息  
得到Mask $QK^T$之后的在Mask $QK^T$上进行Softmax,每一行的和都为1.但是单词0在单词1,2,3,4上的attention score都为0  
第四步:  
使用Mask $QK^T$与矩阵V相乘,得到输出Z,则单词1的输出向量$Z_1$是只包含单词1信息的  
第五步:  
通过上述步骤就可以得到一个Mask Self-Attention的输出矩阵$Z_i$,然后通过Multi-Head Attention拼接多个输出$Z_i$然后计算  
得到第一个Multi-Head Attention的输出Z,Z与输入X维度一样  

第二个Multi-Head Attention  
Decoder block第二个Multi-Head Attention变化不大,主要区别在于Self-Attention的K,V矩阵不是使用  
上一个Decoder block的输出计算的,而是使用Encoder的编码信息矩阵C计算的  
根据Encoder的输出C计算得到K,V,根据上一个Decoder block的输出Z计算Q(第一个Decoder block则使用输入矩阵X进行计算)  
这样做的好处是在Decoder的时候,每一位单词都可以利用到Encoder所有单词的信息  

Softmax预测输出单词  
Softmax根据输出矩阵的每一行愚蠢下一个单词  

Transformer与RNN不同,可以比较好地并行训练  
Transformer本身是不能利用单词的顺序信息的,因此需要在输入中添加位置Embedding,否则Transformer就是一个词袋模型了  
Transformer的重点是Self-Attention结构,其中用到的Q,K,V矩阵通过输出进行线性变换得到  
Transformer中Multi-Head Attention中有多个Self-Attention,可以捕获单词之间多种维度上的相关系数attention score.  












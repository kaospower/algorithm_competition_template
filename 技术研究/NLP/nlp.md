Part I:Basis 
# 2.Introduction  
Logistic regression algorithms are particularly useful because they are easy to train and provide you with  
a good baseline result.  
# 3.supervised machine learning and sentiment analysis(监督学习和情感分析)  
In supervised machine learning you have input features X and a set of labels Y.  
Your goal is to minimize your error rates or cost as much as possible.  
You're going to run your prediction function which takes in parameters data to  
map your features to output labels $\hat Y$.  
The best mapping from features to labels is achieved when the difference between the 
expected values Y and the predicted values $\hat Y$ is minimized.  
Which the cost function does by comparing how closely your output $\hat Y$ is to your label Y.  
Then you can update your parameters and repeat the whole process until your cost is minimized.  

Sentiment analysis using logistic regression  
1.process the raw tweets in your training sets and extract useful features.  
2.train your logistic regression classifier while minimizing the cost.  
3.make your predictions.  
# 4.how to represent a text as a vector  
Feature extraction  
Problems with sparse representations:  
1.Large training time  
2.Large prediction time  
# 5.Positive and negative frequencies(正负频率)  
Frequency dictionary:which maps a word and the class to the number of times that word showed up in 
the corresponding class.
# 6.Feature extraction using positive and negative frequencies  
Encode a tweet or specifically represented as a vector of dimension 3.  
In doing so,you'll have a much faster speed for your logistic regression classifier.  
freqs:dictionary mapping from(word,class) to frequency  
$X_m=[1,\sum\limits_w freqs(w,1),\sum\limits_w freqs(w,0)]$  
$X_m:$Features of tweet m.  
The first feature would be a bias unit equal to 1.  
The second is the sum of the positive frequencies for every unique word on the tweet.  
The third is the sum of the negative frequencies for every unique word on the tweet.  
# 7.Preprocessing  
stemming(词干提取) and stop words(停用词)  
stemming in NLP:simply transforming any word to its base stem(词干),which you could define as the 
set of characters that are used to construct the word and its derivatives(构造词及其派生词的字符集).  
# 8.Integrate everything  
1.build the frequencies dictionary.  
2.initialize the matrix X to match your number of tweets.  
3.deleting stop words,stemming,deleting URLs,and handles and lower casing.  
4.extract the features by summing up the positive and negative frequencies of the tweets.  
# 9.overview of logistic regression  
$h(x^{(i)},\theta)=\frac{1}{1+e^{-\theta^Tx^{(i)}}}$  
# 10.logistic regression training  
gradient descent.  
1.initialize your parameters vector $\theta$  
2.use the logistic function to get values for each of your observations.  
$h=h(X,\theta)$  
3.calculate the gradients of your cost function and update your parameters  
$\nabla=\frac{1}{m}X^T(h-y)$  
$\theta=\theta-\alpha\nabla$  
4.compute your cost J.  
$J(\theta)$  
# 11.logistic regression testing  
pred=$h(X_{val},\theta)>=0.5$  
$\sum\limits_{i=1}^m\frac{pred^{(i)}==y_{val}^{(i)}}{m}$  
# 12.logistic regression cost function  
$J(\theta)=-\frac{1}{m}\sum\limits_{i=1}^m[y^{(i)}log h(x^{(i)},\theta)+(1-y^{(i)})log(1-h(x^{(i)},\theta))]$  
$-\frac{1}{m}\sum\limits_{i=1}^m:$That indicated that you're going to sum over the cost of each training example.  
$-\frac{1}{m}:$indicating that when combined with the sum,this will be some kind of average.  
$-:$The minus sign ensures that your overall costs will always be a positive number.
# 17.Bayes rule  
$P(X|Y)=P(Y|X)\times \frac{P(X)}{P(Y)}$  
# 18.naïve Bayes  
It's a very good,quick and dirty baseline for many texts classification tasks.  
an example of supervised machine learning.  
It's called naive because this method makes the assumption that the features you're using  
for classification are all independent.  
$\prod\limits_{i=1}^m\frac{P(w_i|pos)}{P(w_i|neg)}$  
This expression is called the Naive Bayes inference condition rule for binary classification(二元分类的朴素贝叶斯推理条件规则).  
# 19.Laplacian smoothing(拉普拉斯平滑)  
a technique you can use to avoid your probabilities being zero.  
$P(w_i|class)=\frac{freq(w_i,class)}{N_{class}}$  
$P(w_i|class)=\frac{freq(w_i,class)+1}{N_{class}+V_{class}}$  
$N_{class}=$frequency of all words in class  
$V_{class}=$number of unique words in class  
# 20.log likelihoods 1  
prior ratio(先验比率):$\frac{P(pos)}{P(neg)}$  
likelihood(似然):$\prod\limits_{i=1}^m\frac{P(w_i|pos)}{P(w_i|neg)}$  
$ratio(w)=\frac{P(w|pos)}{P(w|neg)}$  
$\lambda(w)=log\frac{P(w|pos)}{P(w|neg)}$  
you can use that to reduce the risk of numerical underflow(减少数值下溢的风险).  
朴素贝叶斯分数公式:先验比率*似然  
$t=\frac{P(pos)}{P(neg)}\prod\limits_{i=1}^m\frac{P(w_i|pos)}{P(w_i|neg)}$  
if t>1:positive  
if t<1:negative  
取对数,变成log prior(对数先验)+log likelihood(对数似然)  
$log(\frac{P(pos)}{P(neg)}\prod\limits_{i=1}^m\frac{P(w_i|pos)}{P(w_i|neg)})=log\frac{P(pos)}{P(neg)}+\sum\limits_{i=1}^m log\frac{P(w_i|pos)}{P(w_i|neg)}$
# 21.log likelihoods 2  
$\sum\limits_{i=1}^m log\frac{P(w_i|pos)}{P(w_i|neg)}=\sum\limits_{i=1}^m \lambda(w_i)$  
正值表示推文是正面的,负值表示推文是负面的,0表示推文是中立的   
# 22.train the Naive Bayes classifier  
0.Get or annotate a dataset with positive and negative tweets  
1.Preprocess the tweets:process_tweet(tweet)$\rightarrow [w_1,w_2,w_3,...]$  
Lowercase  
Remove punctuation,urls,names  
Remove stop words  
Stemming  
Tokenize sentences  
2.Compute freq(w,class)  
3.Get P(w|class) P(w|neg) using Laplacian smoothing formula  
4.Get $\lambda(w)$  
$\lambda(w)=log\frac{P(w|pos)}{P(w|neg)}$  
5.Compute log prior=log(P(pos)/P(neg))   
$log prior=log\frac{D_{pos}}{D_{neg}}$  
$D_{pos}=$Number of positive tweets  
$D_{neg}=$Number of negative tweets  
If dataset is balanced,$D_{pos}=D_{neg}$ and log prior=0  
# 23.test the Naive Bayes classifier  
$X_{val},Y_{val}\rightarrow$ Performance on unseen data  
Predict using $\lambda$ and logprior for each new tweet  
Accuracy$\rightarrow\frac{1}{m}\sum\limits_{i=1}^m(pred_i==Y_{val_i})$  
# 24.Application of Naive Bayes  
Sentiment analysis  
Author identification  
Spam filtering  
Information retrieval  
Word disambiguation  
# 25.assumptions underlying the naive bayes method  
Naive Bayes的问题:
Independence:Not true in NLP  
Relative frequency of classes affect the model  
Another issue with naive bayes is that it relies on the distribution of the training data sets.  
# 26.analyze errors  
1.Processing as a source errors  
2.Adversarial attacks(对抗攻击)  
# 29.vector space model  
Vector space models will also allow you to capture dependencies between words(捕获词之间的依赖关系).  
Vector space models are used in information extraction(信息抽取) to answer.  
Vector space models allow you to represent words and documents as vectors,this captures the relative meaning.  
# 30.W/W and W/D design  
Word by Word Design  
The co-occurrence(共现) of two different words is the number of times that they appear in your corpus together  
within a certain word distance k.  
Word by document Design  
You will count the times that words from your vocabulary appear in documents that belong to specific categories.  
# 31.Euclidean distance(欧几里得距离)  
$d(B,A)=\sqrt{(B_1-A_1)^2+(B_2-A_2)^2}$  
Euclidean distance for n-dimensional vectors:  
$d(\vec v,\vec w)=\sqrt{\sum\limits_{i=1}^n(v_i-w_i)^2}\rightarrow Norm of (\vec v-\vec w)(比较的向量之间差异的范数)$

```python
import numpy as np

v=np.array([1,6,8])
w=np.array([0,4,6])
d=np.linalg.norm(v-w)
```
# 32.cosine similarity intuition  
The main advantage of this metric over the Euclidean distance is that it isn't biased by the size difference between the representations.  
# 33.cosine similarity score  
vector norm(向量的范数):$||\vec v||=\sqrt{\sum\limits_{i=1}^n v_i^2}$  
Dot product:$\vec v\cdot\vec w=\sum\limits_{i=1}^n v_i\cdot w_i$  
$cos(\beta)=\frac{\vec v\cdot \vec w}{||\vec v||||\vec w||}$  
$Cosine\propto Similarity$  
Cosine Similarity gives values between 0 and 1  
# 35.visualize and PCA  
PCA:principal components analysis(主成分分析)  
Original Space$\rightarrow$Uncorrelated features$\rightarrow$Dimension reduction  
Visualization to see words relationships in the vector space  
# 36.PCA algorithm  
Eigenvector(特征向量):Uncorrelated features for your data.  
Eigenvalue(特征值):the amount of information retained by each feature.  
The Eigenvectors of the co-variance matrix from your data give directions of uncorrelated features and  
the Eigenvalues are the variants of your data sets in each of those new features.  

Mean Normalize Data:$x_i=\frac{x_i-\mu_{x_i}}{\sigma_{x_i}}$  
Get Covariance Matrix  
Perform SVD(奇异值分解)  
SVD:第一个矩阵包含按列堆叠的特征向量,第二个矩阵在对角线上有特征值  

首先,执行词嵌入矩阵与U矩阵(特征向量矩阵)的前n列之间的点积,n等于你最终想要的维度数量    
$X'=XU[:,0:2]$  
Percentage of Retained Variance=$\frac{\sum\limits_{i=0}^1S_{ii}}{\sum\limits_{j=0}^dS_{jj}}$  

# 40.word vector transformation  
$XR\approx Y$  
Loss=$||XR-Y||_F$  
梯度下降:  
$g=\frac{d}{dR}Loss$ gradient  
$R=R-\alpha g$ update  
$$
\begin{pmatrix}
2 & 2\\
2 & 2\\
\end{pmatrix}
$$  
$||A_F||=\sqrt{2^2+2^2+2^2+2^2}=4$  
Frobenius norm:$||A||_F=\sqrt{\sum_\limits{i=1}^m\sum\limits_{j=1}^n|a_{ij}|^2}$  
实际使用Frobenius norm的平方更容易  
$Loss=||XR-Y||_F^2$  
$g=\frac{d}{dR}Loss=\frac{2}{m}(X^T(XR-Y))$  
# 41.KNN  
K-nearest neighbors,for closest matches  
Hash tables  
# 42.Hash tables and Hash functions   
Create a basic hash table  
```python
def basic_hash_table(value_l,n_buckets):
    def hash_function(value,n_buckets):
        return int(value)%n_buckets  
    hash_table={i:[] for i in range(n_buckets)}
    for value in value_l:
        hash_value=hash_function(value, n_buckets)
        hash_table[hash_value].append(value)
    return hash_table
```
# 43.Locality sensitive hashing(局部敏感哈希)  
Which side of the plane?  
```python
def side_of_plance(P,v):
    dotproduct=np.dot(P,v.T)  
    sign_of_dot_product=np.sign(dotproduct)  
    sign_of_dot_product_scalar=np.asscalar(sign_of_dot_product)  
    return sign_of_dot_product_scalar
```
# 44.Mutiple planes(多平面)  
$sign_i\geq 0\rightarrow h_i=1$  
$sign-i<0\rightarrow h_i=0$  
$hash=\sum_i\limits^H 2^i\times h_i$  
Multiple planes,single hash value  
```python
def hash_multiple_plane(P_l,v):
    hash_value=0
    for i,P in enumerate(P_l):
        sign=side_of_plane(P,v)
        hash_i=1 if sign>=0 else 0
        hash_value+=2**i*hash_i
    return hash_value
```
# 45.approximate nearest neighbors(近似最近邻)  
Make one set of random planes  
```python
num_dimensions=2  #300 in assignment
num_planes=3 #10 in assignment
#创建矩阵
random_planes_matrix=np.random.normal(size=(num_planes,num_dimensions))  
v=np.array([[2,2]])  
def side_of_plane_matrix(P,v):
    dotproduct=np.dot(P,v.T)
    sign_of_dot_product=np.sign(dotproduct)
    return sign_of_dot_product
num_planes_matrix=side_of_plane_matrix(random_planes_matrix,v)
```
locality sensitive hashing allows two compute k nearest neighbors,much faster than naive search.  
# 46.Document Search(文档搜索)  
text can be embedded into vector spaces so that nearest neighbors refer to text with similar meaning.  
找到每个单独单词的词向量,然后将它们相加,所有这些词向量的总和成为一个与词向量具有相同维度的文档向量.  
Part II:probabilistic models and how to use them to predict word sequences  
# 0.Introduction  
auto-correction(自动纠错)  
web search suggestions(网站搜索建议)  
# 2.overview  
autocorrect  
# 3.autocorrect  
Autocorrect is an application that changes misspelled words into the correct ones.  
How it works?  
1.Identify a misspelled word.  
2.Find strings n edit distance away.  
3.Filter candidates.  
4.Calculate word probabilities.  
# 4.Building the model 1  
```python
if word not in vocab:
    misspelled=True
```
Edit:an operation performed on a string to change it  
Given a string find all possible strings that are n edit distance away using  
Insert(插入)   
Delete(删除)   
Switch(相邻交换)  
Replace(替换)  
# 5.Building the model 2  
Calculate word probabilities  
$P(w)=\frac{C(w)}{V}$  
$P(w):$Probability of word  
$C(w):$Number of times the word appears    
$V:$Total size of the corpus   
选择概率最高的单词作为自动纠错的替换词  
# 6.Minimum Edit Distance(最小编辑距离)  
用于拼写纠正,文档相似度,机器翻译,DNA测序  
# 7.Minimum Edit Distance Algorithm  
$D[i,j]=source[:I]\rightarrow target[:j]$  
# 8.Minimum Edit Distance Algorithm II  
$$
D[i,j]=
min
\begin{cases}
D[i-1,j]+del_cost\\
D[i,j-1]+ins_cost\\
D[i-1,j-1]+
\begin{cases}
rep_cost,if~src[i]\ne tar[j]\\
0,if~src[i]=tar[j]
\end{cases}
\end{cases}
$$
# 9.Minimum Edit Distance Algorithm III  
Levenshtein distance  
Backtrace  
Dynamic programming  
# 12.parts of speech tagging(词性标注)  
part of speech(词性)  
Part of speech(POS) tagging  

Applications of POS tagging  
make assumptions about semantics  
identifying named entities  
coreference resolution(共指消解)  
speech recognition(语音识别)  
# 13.Markov chains(马尔可夫链)  
Markov chains are a type of stochastic model that describes a sequence of possible events.  
使用有向图来表示马尔可夫链  
$Q={q_1,q_2,q_3}$  
# 14.Markov chains and parts of speech tags  
transition probabilities(转移概率)  
Markov property  
transition matrix(转移矩阵)  
在转移矩阵中,每一行中的所有转移概率应加起来为1  
initial state

states:$Q=\left\{q_1,...,q_N\right\}$  
Transition matrix:  
$$
\begin{pmatrix}
a_{1,1} & \cdots & a_{1,N}\\
\vdots & \ddots & \vdots\\
a_{N+1,1} & \cdots & a_{N+1,N}
\end{pmatrix}
$$
# 15.hidden Markov models(隐马尔可夫模型)  
The name Hidden Markov model implies that states are hidden or not directly observable.
emission probabilities(发射概率)  
Emission matrix
$$
B=
\begin{pmatrix}
b_{11} & \cdots & b_{1V}\\
\vdots & \ddots & \vdots\\
b_{N1} & \cdots & b_{NV}
\end{pmatrix}
$$
Part III:NLP with sequence models  
Part IV:NLP with attention models  
# 16.computing probabilities(计算概率)  
Transition probabilities  
1.Count occurrences of tag pairs  
$C(t_{i-1},t_i)$  
2.Calculate probabilities using the counts  
$P(t_i|t_{i-1})=\frac{C(t_{i-1},t_i)}{\sum\limits_{j=1}^NC(t_{i-1},t_j)}$  
# 17.populate transition matrix(填充转移矩阵)  
smoothing(平滑)  
$P(t_i|t_{i-1})=\frac{C(t_{i-1},t_i)+\epsilon}{\sum\limits_{j=1}^NC(t_{i-1},t_j)+N*\epsilon}$  
# 18.populate emission matrix(填充发射矩阵)  
$P(w_i|t_i)=\frac{C(t_i,w_i)+\epsilon}{\sum_\limits{j=1}^VC(t_i,w_j)+N*\epsilon}=\frac{C(t_i,w_i)+\epsilon}{C(t_i)+N*\epsilon}$  
N表示标签总数,V表示词表大小  
# 19.Viterbi algorithm(维特比算法)  
1.Initialization step  
2.Forward pass  
3.Backward pass  
# 20.Viterbi Initialization  
C:$c_{i,1}=\pi*b_{i,cindex(w_1)}=a_{1,i}*b_{i,cindex(w_1)}$  
D:$d_{i,1}=0$  
# 21.Viterbi Forward pass  
用类似动态规划的方式实现  
$c_{i,j}=\mathop{max~}\limits_{k}c_{k,j-1}*a_{k,i}*b_{i,cindex(w_j)}$  
$d_{i,j}=\mathop{argmax~}\limits_{k}c_{k,j-1}*a_{k,i}*b_{i,cindex(w_j)}$  
# 22.Viterbi Backward pass  
如何使用概率矩阵  
如何使用它来创建路径以便为每个单词分配词性标签    
在此步骤中检索给定单词序列的最可能的词性标签  
矩阵D表示最可能生成我们序列的隐藏状态序列  
$s=\mathop{argmax~}\limits_ic_{i,K}$  
该索引处的概率是生成给定单词序列的最可能的隐藏状态序列的概率  
使用对数概率:  
$log(c_{i,j})=\mathop{max~}\limits_k log(c_{k,j-1})+log(a_k,i)+log(b_i,cindex(w_j))$  
# 24.Week Introduction  
complete(自动补全)  
# 25.N-grams overview(N元语法概述)  
Create language model(LM) from text corpus to  
Estimate probability of word sequences  
Estimate probability of a word following a sequence of words  
Apply this concept to autocomplete a sentence with most likely suggestions  

Other Applications  
Speech recognition  
Spelling correction  
Augmentative communication(辅助沟通系统)  
# 26.N-gram language models and probabilities  
An N-gram is a sequence of N words.  

Sequence notation:  
$w_1^m=w_1w_2\cdots w_m$  
$w_1^3=w_1w_2w_3$  
$w_{m-2}^m=w_{m-2}w_{m-1}w_m$  

Unigram probability(一元语法概率)  
$P(w)=\frac{C(w)}{m}$  

Bigram probability(二元语法概率)  
$P(y\mid x)=\frac{C(x,y)}{\sum\limits_w C(x,w)}=\frac{C(x,y)}{C(x)}$  

Trigram probability(三元语法概率)  
$P(w_3|w_1^2)=\frac{C(w_1^2w_3)}{C(w_1^2)}$  
$C(w_1^2w_3)=C(w_1w_2w_3)=C(w_1^3)$  

N-gram probability(N元语法概率)  
$P(w_N|w_1^{N-1})=\frac{C(w_1^{N-1}w_N)}{C(w_1^{N-1})}$  
$C(w_1^{N-1}w_N)=C(w_1^N)$
# 27.Probability of a sequence(序列概率)  
Conditional probability and chain rule reminder  
$P(B|A)=\frac{P(A,B)}{P(A)}\Rightarrow P(A,B)=P(A)P(B|A)$  
$P(A,B,C,D)=P(A)P(B|A)P(C|A,B)P(D|A,B,C)$  
Problem:Corpus almost never contains the exact sentence we're interested in or even its longer subsequences.  

Approximation of sequence probability  
$P(tea|the~teacher~drinks)\approx P(tea|drinks)$  
$P(the~teacher~drinks~tea)=P(the)P(teacher|the)P(drinks|teacher)P(tea|drinks)$  
Markov assumption:only last N words matter  
$Bigram:P(w_n|w_1^{n-1})\approx P(w_n|w_{n-1})$  
$N-gram:P(w_n|w_1^{n-1})\approx P(w_n|w_{n-N+1}^{n-1})$  

Entire sentence modeled with bigram:$P(w_1^n)\approx\prod\limits_{i=1}^nP(w_i|w_{i-1})$  
$P(w_1^n)\approx P(w_1)P(w_2|w_1)\cdots P(w_n|w_{n-1})$  
# 28.beginning and the end of a sentence(句子开始与结束)  
Start of sentence token $\langle s\rangle$  
$P(\langle s\rangle ~the~teacher~drinks~tea)\approx P(the|\langle s\rangle)P(teacher|the)P(drinks|teacher)P(tea|drinks)$  
Start of sentence token $\langle s\rangle$ for N-grams  
$P(w_1^n)\approx P(w_1|\langle s\rangle\langle s\rangle)P(w_2|\langle s\rangle w_1)\cdots P(w_n|w_{n-2}w_{n-1})$  

N-gram model:add N-1 start tokens $\langle s\rangle$  

End of sentence token $\langle /s\rangle$  
Corpus:  
$\langle s\rangle yes~no$  
$\langle s\rangle yes~yes$  
$\langle s\rangle no~no$  
$P(\langle s\rangle ~yes~yes)=$
$P(yes|\langle s\rangle)\times P(yes|yes)=$
$\frac{C(\langle s\rangle yes)}{\sum\limits_wC(\langle s\rangle w)}\times$
$\frac{C(yes~yes)}{\sum\limits_wC(yes~w)}=$
$\frac{2}{3}\times\frac{1}{2}=\frac{1}{3}$  
$N-gram\Rightarrow just~one\langle/s\rangle$  
the teacher drinks tea$\Rightarrow\langle s\rangle\langle s\rangle$the teacher drinks tea $\langle /s\rangle$  
# 29.N-gram language model  
$P(w_n|w_{n-N+1}^{n-1})=\frac{C(w_{n-N+1}^{n-1},w_n)}{C(w_{n-N+1}^{n-1})}$  
Count matrix  

Probability matrix  
Divide each cell by its row sum  
$sum(row)=\sum\limits_{m \in V}C(w_{n-N+1}^{n-1},w)=C(w_{n-N+1}^{n-1})$  

probability matrix $\Rightarrow$ language model  
Sentence probability  
Next word prediction  

Sentence probability:  
$\langle s\rangle I~learn\langle /s\rangle$  
$P(sentence)=P(I|\langle /s\rangle)P(learn|I)P(\langle /s\rangle|learn)=1\times 0.5\times 1=0.5$  

Log probability  
$P(w_1^n)\approx\prod\limits_{i=1}^nP(w_i|w_{i-1})$  
All probabilities in calculation $\leq$ 1 and multiplying them brings risk of underflow.  

Generative Language model  
Algorithm:  
1.Choose sentence start  
2.Choose next bigram starting with previous word  
3.Continue until $\langle/s\rangle$ is picked  

















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
$X_m=[1,\displaystyle\sum_w freqs(w,1),\displaystyle\sum_w freqs(w,0)]$  
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
$\displaystyle\sum_{i=1}^m\frac{pred^{(i)}==y_{val}^{(i)}}{m}$  
# 12.logistic regression cost function  
$J(\theta)=-\frac{1}{m}\displaystyle\sum_{i=1}^m[y^{(i)}log h(x^{(i)},\theta)+(1-y^{(i)})log(1-h(x^{(i)},\theta))]$  
$-\frac{1}{m}\displaystyle\sum_{i=1}^m:$That indicated that you're going to sum over the cost of each training example.  
$-\frac{1}{m}:$indicating that when combined with the sum,this will be some kind of average.  
$-:$The minus sign ensures that your overall costs will always be a positive number.
# 17.Bayes rule  
$P(X|Y)=P(Y|X)\times \frac{P(X)}{P(Y)}$  
# 18.naïve Bayes  
It's a very good,quick and dirty baseline for many texts classification tasks.  
an example of supervised machine learning.  
It's called naive because this method makes the assumption that the features you're using  
for classification are all independent.  
$\displaystyle\prod_{i=1}^m\frac{P(w_i|pos)}{P(w_i|neg)}$  
This expression is called the Naive Bayes inference condition rule for binary classification(二元分类的朴素贝叶斯推理条件规则).  
# 19.Laplacian smoothing(拉普拉斯平滑)  
a technique you can use to avoid your probabilities being zero.  
$P(w_i|class)=\frac{freq(w_i,class)}{N_{class}}$  
$P(w_i|class)=\frac{freq(w_i,class)+1}{N_{class}+V_{class}}$  
$N_{class}=$frequency of all words in class  
$V_{class}=$number of unique words in class  
# 20.log likelihoods 1  
prior ratio(先验比率):$\frac{P(pos)}{P(neg)}$  
likelihood(似然):$\displaystyle\prod_{i=1}^m\frac{P(w_i|pos)}{P(w_i|neg)}$  
$ratio(w)=\frac{P(w|pos)}{P(w|neg)}$  
$\lambda(w)=log\frac{P(w|pos)}{P(w|neg)}$  
you can use that to reduce the risk of numerical underflow(减少数值下溢的风险).  
朴素贝叶斯分数公式:先验比率*似然  
$t=\frac{P(pos)}{P(neg)}\displaystyle\prod_{i=1}^m\frac{P(w_i|pos)}{P(w_i|neg)}$  
if t>1:positive  
if t<1:negative  
取对数,变成log prior(对数先验)+log likelihood(对数似然)  
$log(\frac{P(pos)}{P(neg)}\displaystyle\prod_{i=1}^m\frac{P(w_i|pos)}{P(w_i|neg)})=log\frac{P(pos)}{P(neg)}+\displaystyle\sum_{i=1}^m log\frac{P(w_i|pos)}{P(w_i|neg)}$
# 21.log likelihoods 2  
$\displaystyle\sum_{i=1}^m log\frac{P(w_i|pos)}{P(w_i|neg)}=\displaystyle\sum_{i=1}^m \lambda(w_i)$  
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
Accuracy$\rightarrow\frac{1}{m}\displaystyle\sum_{i=1}^m(pred_i==Y_{val_i})$  
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
$d(\vec v,\vec w)=\sqrt{\displaystyle\sum_{i=1}^n(v_i-w_i)^2}\rightarrow Norm of (\vec v-\vec w)(比较的向量之间差异的范数)$

```python
import numpy as np

v=np.array([1,6,8])
w=np.array([0,4,6])
d=np.linalg.norm(v-w)
```
# 32.cosine similarity intuition  
The main advantage of this metric over the Euclidean distance is that it isn't biased by the size difference between the representations.  
# 33.cosine similarity score  
vector norm(向量的范数):$||\vec v||=\sqrt{\displaystyle\sum_{i=1}^n v_i^2}$  
Dot product:$\vec v\cdot\vec w=\displaystyle\sum_{i=1}^n v_i\cdot w_i$  
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
Part II:probabilistic models and how to use them to predict word sequences  
Part III:NLP with sequence models  
Part IV:NLP with attention models  
















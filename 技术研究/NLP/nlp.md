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
Part II:probabilistic models and how to use them to predict word sequences  
Part III:NLP with sequence models  
Part IV:NLP with attention models  












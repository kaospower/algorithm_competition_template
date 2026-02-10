# 8.Birthday Problem
Probability that no two people have the same birthday. 
If you are ever in a party with 23 people,you can bet that there's a repeated birthday(50%).  
# 11.Bayes theorem
$P(A|B)=\frac{P(A\cap B)}{P(B)}$  
$P(A\cap B)=P(A)\cdot P(B|A)=P(B)\cdot P(A|B)$  
Bayes formula:  
$P(A|B)=\frac{P(A)\cdot P(B|A)}{P(A)\cdot P(B|A)+P(A')\cdot P(B|A')}$  
prior(先验概率):在不知道任何信息的情况下计算出来的原始概率,P(A)  
posterior(后验概率):事件E给你关于概率的信息,有了这些信息,可以计算出后验概率 P(A|E)  
后验概率总是比先验概率更好的估计,因为我们有了那个提供信息的事件  
naïve Bayes(朴素贝叶斯):所有朴素贝叶斯分类器都假定样本每个特征与其他特征都不相关.
argmax:让某个函数取最大值时，那个“自变量”是什么  
$A\propto B\iff A=k\cdot B$,即A与B成正比  
# Probability mass function(PMF)  
For discrete random variables  
It's defined as the probability that the random variable takes at a particular value.  
# Probability density function(PDF)
For continuous random variables  
It represents the rate at which you accumulate probability around each point.  
# Cumulative Distribution Function(CDF)
Cumulative probability tells you what is the probability that an event happened before  
some reference point.  
$CDF(x)=P(X<=x)$   
对于离散变量,CDF表现为跳跃,  
对于连续变量,CDF表现为平滑  
# 61.Law of Large Numbers(大数定律)
随着样本大小的增加,样本的平均值趋向于接近整个人口的平均值  
n:number of samples  
$X_i: is the i-th random sample from the population$  
Each $X_i$ are independent and identically distributed(i.i.d)(独立同分布)  
as $n\rightarrow \infty,\frac{1}{n}\sum_{i=1}^nX_i\rightarrow E[X]=\mu_X$  
Under certain conditions:
Sample is randomly drawn  
Sample size must be sufficiently large  
Independent observations  
# 62.Central Limit Theorem
As you increase the number of variables you are adding the distribution of this sum,  
your distribution looks more and more like the Gaussian distribution(高斯分布/正态分布)  
# 65.maximum likelihood estimation(MLE):
MLE:we picked the scenario that made the evidence more likely.  
likelihood(似然):the probability of seeing this data based on the model.  
求$p^8(1-p)^2$的最大值:  
取对数$log(p^8(1-p)^2)=8logp+2log(1-p)$  
Log-likelihood,对数似然  
求导$\frac{d}{dp}(8logp+2log(1-p))=\frac{8}{p}-\frac{2}{1-p}$  
拐点是$\hat p=\frac{8}{10}$,此时概率取得最大值  
MLE伯努利示例,MLE高斯分布示例  
Finding the line that most likely produce a point using maximum  
likelihood is exactly the same as minimizing the least square
error using linear regression.  
# 69.Regularization
Log-loss:ll  
L2 Regularization Error,非常数项的系数的平方和t  
Regularization parameter:$\lambda$  
Regularized error:ll+$\lambda$t



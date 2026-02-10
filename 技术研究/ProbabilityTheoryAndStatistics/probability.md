# 11.Bayes theorem
$P(A|B)=\frac{P(A\cap B)}{P(B)}$  
$P(A\cap B)=P(A)\cdot P(B|A)=P(B)\cdot P(A|B)$  
Bayes formula:  
$P(A|B)=\frac{P(A)\cdot P(B|A)}{P(A)\cdot P(B|A)+P(A')\cdot P(B|A')}$  
prior(先验概率):在不知道任何信息的情况下计算出来的原始概率  

1.maximum likelihood estimation(MLE):
we picked the scenario that made the evidence more likely.  
2.likelihood(似然):the probability of seeing this data based on the model.  
3.求$p^8(1-p)^2$的最大值:  
取对数$log(p^8(1-p)^2)=8logp+2log(1-p)$  
Log-likelihood,对数似然  
求导$\frac{d}{dp}(8logp+2log(1-p))=\frac{8}{p}-\frac{2}{1-p}$  
拐点是$\hat p=\frac{8}{10}$,此时概率取得最大值  
MLE伯努利示例,MLE高斯分布示例  
Finding the line that most likely produce a point using maximum  
likelihood is exactly the same as minimizing the least square
error using linear regression.
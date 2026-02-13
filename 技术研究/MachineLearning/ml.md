# 10 Cost Function(成本函数)  
Model:$f_{w,b}(x)=wx+b$  
w,b:parameters,coefficients,weights  
Squared error cost function:$J(w,b)=\frac{1}{2m}\displaystyle\sum_{i=1}^m(\hat y(i)-y(i))^2$  
# 14.Gradient Descent(梯度下降)  
用来求得可微函数的局部最小值  
Keep changing w,b to reduce J(w,b),until we settle at or near a minimum.  
Simultaneous update:  
$tmp_w=w-\alpha\frac{\partial}{\partial w}J(w,b)$  
$tmp_b=b-\alpha\frac{\partial}{\partial b}J(w,b)$  
$w=tmp_w$  
$b=tmp_b$  
$\alpha:$learning rate,usually a small positive number between 0 and 1.  
The learning rate controls how big a step you take when updating the models parameters w and b. 
# 17.Learning Rate(学习率)  
If the learning rate is too small,then gradient descent will work,but it will be slow.  
If the learning rate is too large,gradient descent may overshoot and may never reach the minimum,  
gradient descent may fail to converge and may even diverge.  
If you're already at the local minimum,gradient descent leaves w unchanged because it just updates  
the new value of w to be the exact same old value of w.  
As we get near a local minimum,gradient descent will automatically take smaller steps,and that's  
because as we approach the local minimum,the derivative automatically gets smaller,and that means  
the update steps also automatically get smaller,even if the learning rate $\alpha$ is kept at some  
fixed value.  
You can use it to try to minimize any cost function J.
# 31.Logistic Regression  
sigmoid function(logistic function)  
$g(z)=\frac{1}{1+e^{-z}}$,0<g(z)<1  
$z\rightarrow \infty,g\rightarrow1$  
$z\rightarrow -\infty,g\rightarrow 0$  
$z=0,g=0.5$  
$f_{\vec w,b}(\vec x)=\frac{1}{1+e^{-(\vec w\cdot \vec x+b)}}$,输出结果在0~1之间  
# 36.The Problem of Overfitting(过拟合问题) 
overfit,high variance  
address overfitting:regularization(正则化)
underfitting(欠拟合),high bias  
generalization(泛化):to make good predictions,even on brand new examples that it has never seen before.  



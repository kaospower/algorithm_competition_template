Part I Machine Learning Basis  
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

Part II Advanced Learning Algorithms  
# 2.Demand Prediction  
neural networks(deep learning algorithms)  
If you were to go to the internet and download the parameters of a neural network that  
someone else had trained and whose parameters that posted on the internet,then to use that  
neural network to make predictions would be called inference.   
`layer`:a grouping of neurons which take as input the same or similar features and that in  
turn output a few numbers together.  
`activations values`  
`input layer`,`hidden layer`,`output layer`
`multi-layer perceptron(多层感知器)`  

# 4.Neural network layer  
$w_1^{[1]},第一层第一个参数,上角标表示第几层,下角标表示该层第几个节点$
# 5.More complex neural networks  
By convention,when we say that a neural network has four layers,that includes all  
the hidden layers and the output layer,but wo don't count the input layer.    
Activation value of layer l,unit(neuron) j:
$a_j^{[l]}=g(\vec w_j^{[l]}\cdot \vec a^{[l-1]}+b_j^{[l]})$(j is activation function)  
# 6.Inference:making predictions(forward propagation)  
`forward propagation`  
# 19.Alternatives to the sigmoid activation  
Linear activation function:$g(z)=z$  
Sigmoid activation function:$g(z)=\frac{1}{1+e^{-z}}$  
ReLU activation function:$g(z)=max(0,z)$  
ReLU:Rectified Linear Unit(线性修正单元)  
Softmax activation function  
# 20.Choosing activation functions  
Output Layer:  
When choosing the activation function to use for your output layer,  
usually depending on what is the label Y you are trying to predict,
there'll be one fairly natural choice.   
Binary classification:Sigmoid  activation function  
Regression:  
Linear activation function:y=+/-  
ReLU activation function:y>=0  

Hidden Layer:  
ReLU activation function is by far the most common choice in how neural networks  
are trained by many,many practitioners today.  
With the one exception that you do use a sigmoid activation function in the output  
layer if you have a binary classification problem.  
reason:  
1.ReLU is a bit faster to compute.  
2.ReLu function kind of goes flat only in one part of the graph,  
whereas the sigmoid activation function,it kind of goes flat in  
two places.It goes flat to the left of the graph, and it goes flat  
to the right of the graph. And if you're using gradient descent to train  
a neural network,then when you have a function that is flat in a lot of  
places,gradient descents will be really slow.  
Using the ReLU activation function can cause your neural network to learn  
a bit faster as well.  
# 21.Why do we need activation functions  
如果在隐藏中不用激活函数,神经网络会等价于一个线性回归,失去了意义  
# 23.Softmax  
$a_j=\frac{e^{z_j}}{\displaystyle\sum_{k=1}^Ne^{z_k}}=P(y=j|\vec x)$   
$a_1+a_2+...+a_N=1$  
The softmax regression model is a generalization of logistic regression.  
Crossentropy loss  
$$
    loss(a_1,...,a_N,y)=
    \begin{cases}
    -loga_1 &\text{if y=1}\\ 
    -loga_2 &\text{if y=2}\\
    \quad\quad\quad\quad\vdots\\
    -loga_N &\text{if y=N}\\
    \end{cases}
$$
# 33.Evaluating a model  
`test error`,`training error`  
# 34.Model selection and training/cross validation/test sets  
split your data into three different subsets,which we're going to call the  
training set,the cross-validation set(validation set/dev set),and the test set.  
`Training error`,`Cross validation error`,`Test error`  
If you have to make decisions about your model,such as fitting parameters or choosing  
the model architecture,such as neural network architecture or degree of polynomial if  
you're fitting linear regression,to make all those decisions only using your training  
set and your cross-validation set,and to not look at the test set at all while you're  
still making decisions regarding your learning algorithm.And it's only after you've  
come up with one model,that's your final model,to only then evaluate it on the test set.   
# 47.Error metrics for skewed datasets  
confusion matrix(混淆矩阵):  
$$
\begin{array}{c|c|c}
& \text{Actual 1} & \text{Actual 0}\\
\hline
\text{Predicted 1} & \text{True positive} & \text{False positive}\\
\text{Predicted 0} & \text{False negative} & \text{True negative}\\
\end{array}
$$
$Precision=\frac{True~positive}{predicted~positive}=\frac{TP}{TP+FP}$  
$Recall=\frac{True~positive}{actual~positive}=\frac{TP}{TP+FN}$  
In general,a learning algorithm with either 0 precision or 0 recall is not a  
useful algorithm.  
# 48.Trading off precision and recall  
平衡精确率和召回律  
harmonic mean of P and R(调和平均数):强调较小值  
$F1~score=\frac{1}{\frac{1}{2}(\frac{1}{Precision}+\frac{1}{Recall})}=2\frac{Precision\cdot Recall}{Precision+Recall}$
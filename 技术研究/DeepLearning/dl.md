Part I:Neural Networks and Deep Learning  
# 0.Welcome  
health care,delivering personalized education,precision agriculture,self-driving cars.  
# 1.What is a Neural Network  
Deep learning:training neural networks,sometimes very large neural networks.   
# 2.Supervised Learning with Neural Networks  
Structured Data:means basically databases of data.  
Unstructured Data:raw audio,images where you might want to recognize what's in the image.  
# 3.Why is deep learning only just now taking off  
We went from having a relatively small amount of data to having often a fairly large amount of data.  
# 6.Binary Classification  
Logistic regression is an algorithm for binary classification.  
n:the dimension of this input feature vector.  
m:the number of training examples.  
# 7.Logistic Regression(逻辑回归)  
$\sigma(z)=\frac{1}{1+e^{-z}}$  
# 8.Logistic Regression cost function  
$\hat y^{(i)}=\sigma(w^Tx^{(i)}+b),where~\sigma(z^{(i)})=\frac{1}{1+e^{-z^{(i)}}}$  
Loss(error) function:to measure how good our output $\hat y$ is when the true label is y.  
$\mathcal{L}(y,\hat y)=-(ylog\hat y+(1-y)log(1-\hat y))$  
cost function:measures how well you're doing on the entire training set.  
$J(w,b)=\frac{1}{m}\displaystyle\sum_{i=1}^m\mathcal{L}(\hat y^{(i)},y^{(i)})$
$=-\frac{1}{m}\displaystyle\sum_{i=1}^m[y^{(i)}log\hat y^{(i)}+(1-y^{(i)})log(1-\hat y^{(i)})]$  
The loss function is applied to just a single training example, and the cost function is the cost
of your parameters.  
# 9.Gradient Descent  
$w=w-\alpha\frac{dJ(w,b)}{dw}$  
$b=b-\alpha\frac{dJ(w,b)}{db}$  
$\alpha$:learning rate,controls how big a step we take on each iteration of gradient descent.  
# 14.Logistic Regression Gradient descent  
$\frac{d\mathcal{L(a,y)}}{da}=-\frac{y}{a}+\frac{1-y}{1-a}=\frac{a-y}{a(1-a)}$  
$\frac{d\mathcal{L(a,y)}}{dz}=\frac{d\mathcal{L(a,y)}}{da}\frac{da}{dz}=\frac{a-y}{a(1-a)}a(1-a)=a-y$  
# 15.Gradient descent on m examples  
In the deep learning era,vectorization that is getting rid of for loops has become really important.  
# 16.Vectorization  
$z=w^Tx+b$,w and x are column vectors.  
```python
#python实现
z=np.dot(w,x)+b 
```
Whenever possible,avoid using explicit loops.  
# 17.More vectorization examples  
对列向量v的每一个维度,进行指数运算操作  
```python
import numpy as np
u=np.exp(v)
```

```python
import numpy as np
#NumPy built-in function
np.log(v)  # 逐元素计算对数
np.abs(v)  # 逐元素计算绝对值
np.maximum(v,0) #逐元素计算和0的最大值
v**2 #逐元素计算平方
1/v #逐元素计算倒数
```

```python
# 使用向量化计算逻辑回归
import numpy as np
dw = np.zeros((n-x,1))
```
# 18.Vectorizing Logistic Regression  
X:nxm矩阵,维度是n,样本数量是m  
$z=w^Tx+b$  
$Z=[z^{(1)},z^{(2)},...,z^{(m)}]=w^TX+[b,b,...,b]$
```python
import numpy as np
#NumPy广播机制,自动扩展维度计算,不复制数据，就能让不同形状的数组进行逐元素运算
Z = np.dot(w.T,X)+b #python自动将b扩展成1xm行向量
```
# 19.Vectorizing Logistic Regression's Gradient Computation  
dZ=A-Y  
$db=\frac{1}{m}\displaystyle\sum_{i=1}^mdz^{(i)}=\frac{1}{m}np.sum(dZ)$  
$dw=\frac{1}{m}Xdz^T$  

Implementing Logistic Regression  
迭代次数仍需用for循环实现,其他部分可以用矩阵运算来代替for循环  
$Z=W^TX+b=np.dot(W.T,X)+b$  
$A=\sigma(Z)$  
$dZ=A-Y$  
$dw=\frac{1}{m}XdZ^T$  
$db=\frac{1}{m}np.sum(dZ)$  
$w=w-\alpha dw$  
$b=b-\alpha db$  
# 20.Broadcasting in Python  
```python
cal=A.sum(axis=0) #python矩阵垂直求和
cal=A.sum(axis=1) #python矩阵水平求和
percentage=100*A/(cal.reshape(1,4))
```
If you have an mxn matrix and you add or subtract or multiply or divide with a 1xn matrix,  
then this will copy it m times into an mxn matrix,and then apply the addition,subtraction,  
multiplication or division element wise.  
# 21.A note on python/numpy vectors  
do not use rank 1 arrays.Such as (5,)  
```python
a=np.random.randn(5,1) #5x1行向量
assert(a.shape==(5,1)) #断言语句有助于debug
a=a.reshape((5,1)) #将秩为1的数组转化成向量
```
# 22.Quick tour of Jupyter/ipython notebooks
```python
% matplotlib inline #让 matplotlib 画出来的图 直接显示在 Notebook 输出区域里
```
restart kernel  
# 26.Neural Network Representation  
When we count layers in neural networks,we don't count the input layer.  
# 27.Computing a Neural Network's Output  
$z^{[1]}=W^{[1]}x+b^{[1]}$  
$a^{[1]}=\sigma(z^{[1]})$  
$z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}$  
$a^{[2]}=\sigma(z^{[2]})$
# 29.Explanation for vectorized implementation  
for i=1 to m  
$z^{[1](i)}=W^{[1]}x^{(i)}+b^{[1]}$  
$a^{[1](i)}=\sigma(z^{[1](i)})$  
$z^{[2](i)}=W^{[2]}a^{[1](i)}+b^{[2]}$  
$a^{[2](i)}=\sigma(z^{[2](i)})$

$Z^{[1]}=W^{[1]}X+b^{[1]}$  
$A^{[1]}=\sigma(Z^{[1]})$  
$Z^{[2]}=W^{[2]}A^{[1]}+b^{[2]}$  
$A^{[2]}=\sigma(Z^{[2]})$  
# 30.Activation functions  
sigmoid function:$a=\frac{1}{1+e^{-z}}$  never use this except for the output layer if you are doing binary classification.     
tanh function(hyperbolic tangent function):$a=\frac{e^z-e^{-z}}{e^z+e^{-z}}$  
ReLU:$a=max(0,z)$  
Leaky ReLu:$a=max(0.01z,z)$  
# 32.Derivatives of activation functions 
Sigmoid activation function  
$a=g(z)=\frac{1}{1+e^{-z}}$  
$\frac{d}{dz}g(z)=\frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}})=g(z)(1-g(z))=a(1-a)$  
Tanh activation function  
$g(z)=tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$  
$\frac{d}{dz}g(z)=1-(tanh(z))^2$  
ReLU and Leaky ReLU  
ReLU  
g(z)=max(0,z)  
$$
g'(z)=\begin{cases}  
0 &\text{if z<0}\\
1 &\text{if z>=0}\\
\end{cases}  
$$
Leaky ReLU  
g(z)=max(0.01z,z)
$$  
g'(z)=\begin{cases}  
0.01 &\text{if z<0}\\
1 &\text{if z>=0}\\
\end{cases}
$$
# 33.Gradient descent for neural networks  
Formulas for computing derivatives  
Forward propagation:  
$Z^{[1]}=w^{[1]}X+b^{[1]}$  
$A^{[1]}=g^{[1]}(Z^{[1]})$  
$Z^{[2]}=w^{[2]}A^{[1]}+b^{[2]}$  
$A^{[2]}=g^{[2]}(Z^{[2]})=\sigma(Z^{[2]})$  
Backpropagation(cost function对相应变量求导):  
$dZ^{[2]}=A^{[2]}-Y$  
$dw^{[2]}=\frac{1}{m}dZ^{[2]}A^{[1]T}$  
$db^{[2]}=\frac{1}{m}np.sum(dZ^{[2]},axis=1,keepdims=True)$  
$dZ^{[1]}=W^{[2]T}dZ^{[2]}*g^{[1]'}(Z^{[1]})$  
$dw^{[1]}=\frac{1}{m}dZ^{[1]}X^T$  
$db^{[1]}=\frac{1}{m}np.sum(dZ^{[1]},axis=1,keepdims=True)$  
# 34.Backpropagation intuition(Optional)  
We don't need to take derivatives with respect to the input X,  
since the input X for supervised learning is fixed,so we're not  
trying to optimize X,so we won't bother to take derivatives,
at least for supervised learning with respect to X.  
# 35.Random Initialization  
$w^{[1]}=np.random.randn((2,2))*0.01$  
$b^{[1]}=np.zeros((2,1))$  
如果初始权重过大,就会处于tanh或者sigmoid函数的平坦部分,梯度下降就会异常缓慢  
# 39.Getting your matrix dimensions right  
$w^{[L]}:(n^{[L]},n^{[L-1]})$  
$b^{[L]}=(n^{[L]},1)$  
$dw^{[L]}=(n^{[L]},n^{[L-1]})$  
$db^{[L]}=(n^{[L]},1)$  
# 40.Why deep representations  
There are functions you can compute with a "small" L-layer deep neural network  
that shallower networks require exponentially more hidden units to compute.  
# 42.Forward and backward propagation  
Forward propagation for layer l  
Input $a^{[l-1]}$  
Output $a^{[l]},cache(z^{[l]})$  
$Z^{[L]}=W^{[L]}\cdot A^{[L-1]}+b^{[L]}$  
$A^{[L]}=g^{[L]}(Z^{[L]})$  
Backward propagation for layer l  
$dZ^{[L]}=dA^{[L]}*g^{[L]'}(Z^{[L]})$  
$dW^{[L]}=\frac{1}{m}dZ^{[L]}\cdot A^{[L-1]T}$  
$db^{[L]}=\frac{1}{m}np.sum(dZ^{[L]},axis=1,keepdims=True)$  
$dA^{[L-1]}=W^{[L]T}\cdot dZ^{[L]}$

# 43.Parameters vs Hyperparameters  
Hyperparameters:  
learning rate $\alpha$(学习率)  
iterations   
hidden layers(隐藏层的数量)  
hidden units(每个隐藏层节点个数)  
choice of activation function  
momentum term  
min-batch size  
various forms of regularization parameters  
# 44.What does this have to do with the brain  
关系不大  
Part II:Improving Deep Neural Networks:Hyperparameter tuning,Regularization and Optimization  
# 0.Train/dev/test sets  
layers,hidden units,learning rates,activation functions  
training set,hold-out cross validation set/development set,test set.  
The workflow is that you keep on training algorithms on your training set,  
use your dev set to see which of many different models performs best on your  
dev set,and then after having done this long enough,when you have a final model  
that you want to evaluate,you can take the best model you have found and evaluate  
it on your test set in order to get an unbiased estimate of how well your algorithm is doing.  
60/20/20,widely considered best practice in machine learning.(less than 10000 examples)   
In the modern big data era,the trend is that your dev and test sets have becoming a much  
smaller percentage of the total.(1000000 examples)  
98/1/1  
Not having a test set might be okay(only dev set).  
# 1.Bias/Variance  
underfitting:high bias  
overfitting:high variance  
Bayes error  
under the assumption that the Bayes error is quite small and that your training and  
your dev sets are drawn from the same distribution.  
# 2.Basic "recipe" for machine learning  
high bias:try a bigger network,more hidden layers or more hidden units,train it longer,  
run training set longer,try some more advanced optimization algorithms.  
high variance:more data,regularization,find a more appropriate neural network architecture.
# 3.Regularization  
Logistic regression  
$J(w,b)=\frac{1}{m}\displaystyle\sum_{i=1}^m\mathcal L(\hat y^{(i)},y^{(i)})+\frac{\lambda}{2m}||w||^2_2$  
L_2 regularization:$||w||^2_2=\displaystyle\sum_{j=1}^{n_x}w_j^2=w^Tw$  
L_1 regularization:$\frac{\lambda}{2m}\displaystyle\sum_{j=1}^{n_x}|w_j|=\frac{\lambda}{2m}||w||_1$  
$\lambda$,the regularization parameter,hyperparameter.  
If you use L1 regularization,then W will end up being sparse.  
And what that means is that the W vector will have a lot of 0s in it.  
Neural network  
$J(w^{[1]},b^{[1]},...,w^{[l]},b^{[l]})=\frac{1}{m}\displaystyle\sum_{i=1}^m\mathcal{L}(\hat y^{(i)},y^{(i)})$  
Frobenius norm of a matrix:$||w^{[l]}||^2_F=\displaystyle\sum_{i=1}^{n^{[l-1]}}\displaystyle\sum_{j=1}^{n^{[l]}}(w_{ij}^{[l]})^2$    
L2 regularization is sometimes also called weight decay.  
$(1-\frac{\alpha\lambda}{m})$  
# 4.Why regularization reduces overfitting  
1.$\lambda$big,set W close to zero,set the weights to be close to zero,  
that it's basically zeroing out a lot of the impact of these hidden units.  
So you end up with what might feel like a simpler network.  
2.If the regularization parameter is very large,the parameter w is very small,  
so z will be relatively small,kind of ignoring the effects of b for now.So z will be  
relatively small and takes on a small range of values.And so the activation function  
tanh will be relatively linear.And so your whole neural network will be computing something  
not too far from a big linear function,which is therefore a pretty simple function,rather than  
a very complex highly nonlinear function.   
# 5.Dropout regularization  
Inverted dropout
```python
import numpy as np
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep.prob
a3 = np.multiply(a3,d3)
a3/=keep.prob
```
no matter what you set the keep prob to,it ensures that the expected value of  
A3 remains the same.   
# 6.Understanding dropout  
Dropout randomly knocks out units in your network.So it's as if on every iteration,
you're working with a smaller neural network.And so using a smaller neural network  
seems like it should have a regularizing effect.  
Can't rely on any one feature,so have to spread out weights.Shrink weight.  
If you're more worried about some layers overfitting than others,  
you can set a lower keep prop for some layers than others.  
Dropout is a regularization technique.It helps prevent overfitting.   
In computer vision,people often use drop out.  
# 7.Other regularization methods  
Early stopping:it looks like your neural network was doing best around that iteration.  
So we're just going to stop training your neural network halfway and take whatever value  
achieved this dev set error.  
The main downside of early stopping is that this couples these two tasks.  
The real advantage of early stopping is that running the gradient descent process just  
once,you get to try out values of small w,midsize w,and large w without needing to try  
a lot of values of the L2 regularization hyperparameter lambda.  
# 8.Normalizing inputs  
$x:=x-\mu$   
$x/=\sigma$  
That just makes your cost function J easier and faster to optimize.  
通过正则化$\mu=0,\sigma=1$  
# 9.Vanishing/exploding gradients(梯度消失/爆炸)   
When you're training a very deep network,your derivatives can sometimes  
get either very,very big or very,very small,maybe even exponentially small,  
and this makes training difficult.  
# 10.Weight initialization for deep networks  
$var(w)=\frac{2}{n}$  
ReLU activation function:$w^{[l]}=np.random.randn(shape)*np.sqrt(\frac{2}{n^{[l-1]}})$  
tanh activation function:$w^{[l]}=np.random.randn(shape)*np.sqrt(\frac{1}{n^{[l-1]}})$  
It definitely helps reduce the vanishing and exploding gradients problem.  
Xavier initialization.  
# 15.Mini-batch gradient descent  
mini-batch  
one epoch of training:a single pass through the training set.  
When you have a large training set,mini-batch gradient descent runs much faster  
than batch gradient descent.  
# 16.Understanding mini-batch gradient descent  
If mini-batch size=m,Batch gradient descent.  
You're processing a huge training set on every iteration.  
So the main disadvantage of this if that it takes too much time,  
too long per iteration.  
If mini-batch size=1,stochastic gradient descent.  
stochastic gradient descent can be extremely noisy,
and on average it'll take you in a good direction,  
but sometimes it'll head in the wrong direction as well.  
And stochastic gradient descent won't ever converge(永不收敛).  
总是在最小值附近振荡徘徊  
You lose almost all you speed up from vectorization.  

If you have a small training set,just use batch gradient descent.(m<=2000)  
If you have a bigger training set,mini-batch size:64,128,256,512  
You code runs faster if your mini-batch size is a power of 2.  
fit in CPU/GPU memory.  
The mini-batch size is actually another hyperparameter.  
# 17.Exponentially weighted averages(指数加权平均)  
exponentially weighted averages/exponentially weighted moving average.  
$v_t=\beta v_{t-1}+(1-\beta)\theta_t$  
$v_t$ as approximately averaging over $\approx \frac{1}{1-\beta}$days' temperature.  
# 18.Understanding exponentially weighted averages  
$v_\theta:=\beta v_\theta+(1-\beta)\theta_t$  
# 19.Bias correction in exponentially weighted average  
$\frac{v_t}{1-\beta^{t}}$  
As t becomes large,$\beta^t$ will approach 0,the bias correction makes almost no difference.  
During initial phase of learning,bias correction can help you to obtain  
a better estimate of the temperature.  
# 20.Gradient descent with momentum(动量梯度下降)  
There's an algorithm called momentum or gradient descent with momentum,  
that almost always works faster than the standard gradient descent algorithm.  

$v_{dw}=0,v_{db}=0$    
On iteration t:    
Compute dw,db on current mini-batch.  
$v_{dw}=\beta v_{dw}+(1-\beta)dw$  
$v_{db}=\beta v_{db}+(1-\beta)db$  
$w:=w-\alpha v_{dw}$  
$b:=b-\alpha v_{db}$  
$Hyperparameters:\alpha,\beta$
$\beta=0.9$  

# 21.RMSprop(root mean square propagation,均方根传播)  
On iteration t:  
compute dw,db on current mini-batch  
$S_{dw}=\beta_2 S_{dw}+(1-\beta_2)dw^2$  
$S_{db}=\beta_2 S_{db}+(1-\beta_2)db^2$  
$w:=w-\alpha\frac{dw}{\sqrt{S_{dw}}+\epsilon}$  
$b:=b-\alpha\frac{db}{\sqrt{S_{db}}+\epsilon}$  

You could use a larger learning rate $\alpha$ and get faster learning without diverging  
in the vertical direction.  
# 22.Adam optimization algorithm(Adaptive Moment Estimation,自适应矩估计)  
On iteration t:  
compute dw,db using current mini-batch  
$V_{dw}=\beta_1 V_{dw}+(1-\beta_1)dw$  
$V_{db}=\beta_1 V_{db}+(1-\beta_1)db$  
$S_{dw}=\beta_2 S_{dw}+(1-\beta_2)dw^2$  
$S_{db}=\beta_2 S_{db}+(1-\beta_2)db^2$  
$V_{dw}^{corrected}=V_{dw}/(1-\beta_1^t)$  
$V_{db}^{corrected}=V_{db}/(1-\beta_1^t)$  
$S_{dw}^{corrected}=S_{dw}/(1-\beta_2^t)$  
$S_{db}^{corrected}=S_{db}/(1-\beta_2^t)$  
$W:=W-\alpha\frac{V_{dw}^{corrected}}{\sqrt{S_{dw}^{corrected}}+\epsilon}$  
$b:=b-\alpha\frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected}}+\epsilon}$  
$\alpha$:needs to be tune   
$\beta_1=0.9$  
$\beta_2=0.999$  
$\epsilon=10^{-8}$  
# 23.Learning rate decay  
$\alpha=\frac{1}{1+decay-rate*epoch-num}\alpha_0$  
Other learning rate decay methods  
exponential decay  
$\alpha=0.95^{epoch-num}\cdot \alpha_0$  
$\alpha=\frac{k}{\sqrt{epoch-num}}\cdot \alpha_0$ or $\frac{k}{\sqrt{t}}\alpha_0$  
discrete staircase  
manual decay  
# 24.The problem of local optima(局部最优问题)  
saddle point(鞍点):梯度为0但不一定是最优解,比如在一个方向上是凹的,在其他方向是凸的  
plateau(高原区域):a region where the derivative is close to zero for a long time.  
Unlikely to get stuck in a bad local optima.   
Plateaus can make learning slow.  
# 26.Tuning process  
learning rate $\alpha$  
momentum term $\beta$  
mini-batch size   
hidden units  
layers  
learning rate decay  

choose the points at random,try out the hyperparameters on this randomly chosen set of points.  
coarse-to-fine sampling scheme.(从粗到精的采样方案):zoom in to a smaller region of the hyperparameters,  
and then sample more densely within this space.  
The two key takeaways are use random sampling,not a grid search and consider optionally,  
but consider implementing a coarse-to-fine search process.   
# 27.Using an appropriate scale to pick hyperparameters  
Appropriate scale for hyperparameters  
logarithmic scale.  
r=-4*np.random.rand()  
$\alpha=10^r$  
Hyperparameters for exponentially weighted averages  
use $1-\beta$ and logarithmic scale,it causes you to sample more densely in the regime  
of when beta is close to 1 or alternatively when 1 minus $\beta$ is close to 0.  
# 28.Hyperparameters tuning in practice:Pandas vs.Caviar   
Panda approach(熊猫策略,一次只训练一个模型)  
Babysitting one model:watching a performance,and patiently nudging the learning rate up or down.   
you have a huge dataset,but not a lot of computational resources,not a lot of CPUs and GPUs.  

Caviar approach(鱼子酱策略,并行训练)  
Training many models in parallel  
multiple learning curves.  
# 29.Normalizing activations in a network  
Batch normalization/Batch Norm(批量归一化):it normalizes the mean and variance of these hidden unit values.   
Implementing Batch Norm:  
$\mu=\frac{1}{m}\displaystyle\sum_i z^{(i)}$  
$\sigma^2=\frac{1}{m}\displaystyle\sum_i(z^{(i)}-\mu)^2$  
$z_{norm}^{(i)}=\frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}$  
every component of z has mean 0 and variance 1.  
$\tilde z^{(i)}=\gamma z^{(i)}_{norm}+\beta$  
If $\gamma=\sqrt{\sigma^2+\epsilon},\beta=\mu$,then $\tilde z^{(i)}=z^{(i)}$  
# 30.Fitting Batch Norm into a neural network  
tf.nn.batch-normalization  

Implementing gradient descent  
compute forward propagation on $X^{\left\{t\right\}}$   
In each hidden layer,use BN to replace $z^{[l]}$ with $\tilde z^{[l]}$  
Use backpropagation to compute $dw^{[l]},d\beta^{[l]},d\gamma^{[l]}$  
update parameters  
$w^{[l]}=w^{[l]}-\alpha dw^{[l]}$  
$\beta^{[l]}=\beta^{[l]}-\alpha d\beta^{[l]}$  
$\gamma^{[l]}=...$
# 31.Why does Batch Norm work?  
1.speed learning.  
2.It makes weights later or deeper in the neural network,more robust to changes  
to weights in earlier layers of the neural network.  

covariate shift(协变量偏移)   
What Batch Norm does is it reduces the amount that the distribution of these hidden unit  
values shifts around.  
Batch Norm reduces the problem of the input values changing.   

Batch Norm as regularization  
Each mini-batch is scaled by the mean/variance computed on just that mini-batch.  
This adds some noise to the values $z^{[l]}$ within that mini-batch.So similar to dropout,  
it adds some noise to each hidden layer's activations.  
This has a slight regularization effect.  
# 32.Batch Norm at test time  
What's actually done in order to apply your neural network at test time,  
is to come up with some separate estimate of $\mu$ and $\sigma^2$  
estimate this using an exponentially weighted average.  

Part III:Structuring your machine Learning project  
Part IV:Convolutional Neural Networks(CNN)  
Computer Vision Problems:
Image Classification  
Object Detection  
Neural Style Transfer(神经风格迁移)  
convolution  
vertical edges,horizontal edges  
grayscale matrix(灰度矩阵)  
filter(滤波器)/kernel(卷积核)  
edge detection(边缘检测)  
Part V:Natural Language Processing:Building sequence models  
# 0.Why sequence models?  
sequence data   
speech recognition(语音识别)   
music generation(音乐生成)  
sentiment classification(情感分类)  
DNA sequence analysis(DNA序列分析)   
machine translation(机器翻译)  
video activity recognition(视频活动识别)  
named entity recognition(命名实体识别)  
All of these problems can be addressed as supervised learning with label data x,y  
as the training set.  
# 1.Notation  
建立Vocabulary/dictionary(字典)  
找出出现频率最高的k个词  
one-hot representations to represent each of these words.  
<UNK>:represent where it's not in your vocabulary.  
# 2.Recurrent Neural Network Model  
Why not a standard network?  
1.Inputs,outputs can be different lengths in different examples.  
2.Doesn't share features learned across different positions of text.  

Time zero activation is the most common choice.  
One limitation of this particular neural network structure is that the prediction  
at a certain time uses inputs,or uses information from the inputs earlier in the  
sequence,but not information later in the sequence.  

$a^{<0>}=\vec 0$  
$a^{<1>}=g(w_{aa}a^{<0>}+w_{ax}x^{<1>}+b_a)$  activation function:tanh/ReLU  
$\hat y^{<1>}=g(w_{ya}a^{<1>}+b_y)$  activation function:sigmoid/softmax  

forward propagation:  
$a^{\langle t\rangle}=g(w_{aa} a^{\langle t-1\rangle}+w_{ax}x^{\langle t\rangle}+b_a)$  
$\hat y^{\langle t\rangle}=g(w_{ya}a^{\langle t\rangle}+b_y)$  
简化表示  
$a^{\langle t\rangle}=g(w_a[a^{\langle t-1\rangle},x^{\langle t\rangle}]+b_a)$  
$y^{{\langle t\rangle}}=g(w_y a^{\langle t\rangle}+b_y)$   

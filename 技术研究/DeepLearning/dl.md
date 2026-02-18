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
Part II:Improving Deep Neural Networks:Hyperparameter tuning,Regularization and Optimization  
Part III:Structuring your machine Learning project  
Part IV:Convolutional Neural Networks(CNN)  
Part V:Natural Language Processing:Building sequence models  
Computer Vision Problems:
Image Classification  
Object Detection  
Neural Style Transfer(神经风格迁移)  
convolution  
vertical edges,horizontal edges  
grayscale matrix(灰度矩阵)  
filter(滤波器)/kernel(卷积核)  
edge detection(边缘检测)  
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

# 30.Activation functions  
sigmoid function:$a=\frac{1}{1+e^{-z}}$  never use this except for the output layer if you are doing binary classification.     
tanh function(hyperbolic tangent function):$a=\frac{e^z-e^{-z}}{e^z+e^{-z}}$  
ReLU:$a=max(0,z)$  
Leaky ReLu:$a=max(0.01z,z)$  
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
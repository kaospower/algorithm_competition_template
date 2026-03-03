# 6.Systems of sentences  
Complete,Redundant,Contradictory  
When a system is redundant(冗余) or contradictory(矛盾),it's called a singular system(奇异系统).  
When a system is complete(完备),it's called a non-singular system(非奇异系统).  
A measure of how redundant a system is:rank(秩).  
# 9.A geometric notion of singularity(奇异性几何意义)  
系统中的常数不影响系统奇异或非奇异的判断
# 10.Singular vs nonsingular matrices(奇异和非奇异矩阵)  
矩阵是奇异的或非奇异的,取决于系统是奇异还是非奇异的.
# 11.Linear dependence and independence(线性相关和无关)  
奇异:线性相关  
非奇异:线性无关  
# 12.The determinant(行列式)  
returns zero if the matrix is singular.  
return non-zero if the matrix is nonsingular.  

三阶行列式:主对角线乘积+两条辅主对角线乘积-反对角线乘积-两条辅反对角线乘积  
upper triangular(上三角矩阵):everything below the diagonal is a zero.  
上三角矩阵的行列式等于主对角线元素乘积  
# 14.Solving non-singular system of linear equations  
Multiplying by a constant(乘上常数)  
Adding two equations(将两个方程相加)  
Divide by coefficient of a  
Subtract equation 1 from equation 2  
# 15.Solving singular system of linear equations  
无解或者无穷多解  
# 17.Matrix row reduction/Gaussian elimination(矩阵行简化/高斯消元)   
Row echelon form(行阶梯形式)  
Reduced row echelon form(简化行阶梯形式)  
行阶梯形式:  
主对角线有一排1,接着可能一排0,主对角线下面全是0  
1的右侧可以是任何数字,0的右侧必须是0  
# 18.Row operations that preserve singularity(保持奇异性的行操作)  
row operations(行操作) preserve the singularity of a matrix(保持矩阵的奇异性).  
交换矩阵的两行,行列式变为相反数  
将矩阵的一行乘上一个非0数x,行列式变为原来的x倍  
将矩阵的两行相加作为新的一行,行列式不变   
# 19.Rank of a matrix(矩阵的秩)  
秩在某种程度上衡量了矩阵或其对应线性方程组所包含的信息量.  
Compressing Images:Reducing rank  
像素化的图像是矩阵,矩阵的秩与存储对应图像所需的空间量有关.  
singular value decomposition(SVD,奇异值分解)中有一种技术:在尽可能少地改变矩阵的情况下降低矩阵的秩  
一个句子系统所包含的信息量定义为系统的秩  
矩阵的秩被定义为对应方程组的秩  
点的维度是0,直线的维度是1,平面的维度是2  
对于行为2的矩阵:Rank=2-(解空间的维度)  
对于行为n的矩阵:Rank=n-(解空间的维度)  
秩和解空间维度总是加起来等于矩阵的行数  
矩阵是非奇异的当且仅当矩阵满秩时,即秩等于行数时  
# 21.Row echelon form(行阶梯形)  
矩阵的秩是行阶梯形对角线上1的数量  
当且仅当行阶梯形对角线上只有1没有0时,矩阵是非奇异的  
# 22.Row echelon form:General case(一般奇异矩阵)  
零元都在底部  
主元:每行最左非零元  
每个主元必须严格在其上方行的主元右侧  
矩阵的秩就是主元的数量  
# 23.Reduced row echelon form(简化行阶梯形)  
简化行阶梯形:  
本身是行阶梯形  
每个主元是1  
任何主元上面的元素必须是0  
矩阵的秩是主元的数量  
简化行阶梯形对应于解决方程组  
# 24.The Gaussian Elimination Algorithm(高斯消元法)  
augmented matrix(增广矩阵):将原方程等号右侧的常数值作为一列放在系数矩阵右侧  
back substitution(回代法)  
主对角线全为1,其他位置全为0,称为单位矩阵   
将增广矩阵除了最后一列简化为单位矩阵,就实现了解方程组  
在高斯消元过程中,如果发现某一行为0(不包含最后一个元素),说明矩阵是奇异的,无解或者有无穷多解  
如果该全0行的常数项为0,说明有无穷多解,如果非0,无解  

高斯消元法过程:  
1.创建增广矩阵  
2.将矩阵转化为简化行阶梯形   
3.通过回代找出变量的值(转化成单位矩阵)  
4.如果遇到全0行,停止,因为这个方程组是奇异的  
# 26.machine learning intuition  
vectors,matrices,tensors(向量,矩阵,张量)  
# 27.Vectors and their properties  
taxicab distance(曼哈顿距离)  
$L_1-norm=|(a,b)|_1=|a|+|b|$  
$L_2-norm=|(a,b)|_2=\sqrt{a^2+b^2}$  
一般形式:  
$L_1~norm:||x||_1=|x_1|+|x_2|+...+|x_n|$  
$L_2~norm:||x||_2=\sqrt{x_1^2+x_2^2+...+x_n^2}$  
# 29.The dot product  
$|u|_2=\sqrt{\langle u,u \rangle},\langle u,u \rangle 表示u和u的点积$  
# 30.Geometric dot product(点积几何意义)  
$\langle u,v \rangle=|u||v|cos(\theta)$  
# 31.Multiplying a matrix by a vector(矩阵与向量乘法)  
矩阵和列向量乘法表示方程组  
# 33.Linear transformations as matrices  
基向量被发送的位置,就是矩阵的列  
# 34.Matrix multiplication  
线性变换作用在左边的向量上  
# 35.The identity matrix(单位矩阵)  
对角线上有1,其它地方都是零  
# 36.Matrix inverse(矩阵逆)  
通过求解线性方程组求矩阵逆  
# 37.Which matrices have inverse?(可逆矩阵条件)  
Non-singular matrices always have an inverse(非奇异矩阵总是有逆).  
Singular matrices never have an inverse(奇异矩阵永远没有逆).  
可逆矩阵的行列式非0,不可逆矩阵行列式为0    
# 38.Neural networks and matrices(神经网络与矩阵)  
在原矩阵中增加一个全1列,在模型中增加bias属性,可以使模型实现增加一个bias项  
与运算也可以作为一个感知器(单层神经网络)  
# 40.Introduction  
Principal Component Analysis(主成分分析)  
Reduce dimensions(columns) of dataset  
Preserve as much information as possible  
# 41.Singularity and rank of linear transformations(线性变换奇异性与秩)  
change of basis(基变换)  
如果将其乘以矩阵后覆盖整个平面的点,则变换是非奇异的,反之亦然.  
右侧覆盖的点称为image of the transformation(变换的像).  
线性变换的秩:线性变换的像的维数  
# 42.Determinant as an area(行列式作为面积)  
按逆时针顺序取向量,平行四边形的面积为负  
按顺时针顺序取向量,平行四边形的面积为正  
# 43.Determinant of a product(矩阵乘积的行列式)  
det(AB)=det(A)det(B)  
任何矩阵乘上奇异矩阵变成奇异矩阵  
# 44.Determinant of inverse(逆矩阵的行列式)  
$det(A^{-1})=\frac{1}{det(A)}$  
# 45.Basis(基)  
空间中的每一点都可以表示为基中元素的线性组合  
两个方向相同或相反的向量不构成一个基底  
# 46.Span(线性代数扩张成空间)  
a basis is a minimal spanning set.  
基的length就是平面的维数  
如果一个向量组中的任何向量都不能表示为其他向量的线性组合,那么这组向量就被称为线性独立的  
一个基是满足两个条件的向量集合:  
1.这个集合必须张成一个向量空间  
2.并且集合中的向量必须线性无关   
# 47.Eigenbasis(特征基)  
它将平行四边形映射到另一个边平行于原始平行四边形的平行四边形  
# 48.Eigenvalue and Eigenvectors(特征值和特征向量)  
$Av_1=\lambda_1 v_1$  
$Av_2=\lambda_2 v_2$  
$v_1$和$v_2$是矩阵A的特征向量   
$\lambda_1$和$\lambda_2$是矩阵A的特征值  

1.计算特征基的逆  
2.将该逆乘上向量  

$Av=\lambda v$:沿着该向量的矩阵乘法仅变成标量乘法  
可以把特征向量看作线性变换中拉伸的方向,特征值告诉你它被拉伸了多少  
可以利用特征向量创建一种基,称为特征基  
通常会将特征基表示为一个矩阵,每列包含一个特征向量  
特征向量可以节省工作量,并以强大的方式表征线性变换  
# 49.Calculating Eigenvalues and Eigenvectors(特征值与特征向量计算)  
If $\lambda$ is an eigenvalue:  
$$
\begin{bmatrix}
2 & 1\\
0 & 3
\end{bmatrix}
\begin{bmatrix}
x\\
y
\end{bmatrix}
=\begin{bmatrix}
\lambda & 0\\
0 & \lambda
\end{bmatrix}
\begin{bmatrix}
x\\
y
\end{bmatrix}
$$
$$
\begin{bmatrix}
2-\lambda & 1\\
0 & 3-\lambda
\end{bmatrix}
\begin{bmatrix}
x\\
y
\end{bmatrix}
=\begin{bmatrix}
0\\
0
\end{bmatrix}
$$
$$
det
\begin{bmatrix}
2-\lambda & 1\\
0 & 3-\lambda
\end{bmatrix}
=0
$$
特征多项式:$(2-\lambda)(3-\lambda)-1\cdot 0=0$  
$\lambda=2,\lambda=3$  

特征多项式:$det(A-\lambda I)=0$  
总会有无数个潜在的特征向量位于同一条线上  
行列式只对方阵有定义,对于任何方阵,你都可以找到特征值和特征向量  
如果矩阵不是方阵,它就没有任何特征向量或特征值  
# 50.On the Number of Eigenvectors(特征向量数量分析)  
对于二阶方阵:  
设其特征值为$\lambda_1,\lambda_2$  
如果$\lambda_1\ne\lambda_2$,则有两个特征向量   
如果$\lambda_1=\lambda_2$,则有一个或者两个特征向量  
对于三阶方阵:  
如果$\lambda_1\ne\lambda_2\ne\lambda_3$,则有三个特征向量  
如果$\lambda_1=\lambda_2\ne\lambda_3$,则有两个或三个特征向量  
如果$\lambda_1=\lambda_2=\lambda_3$,则有一个或两个或三个特征向量   
# 51.Dimensionality Reduction and Projection(降维与投影)  
projection(投影):move your data points into a vector space with fewer dimensions  
乘以向量会使点沿该向量投影,并且除以向量的范数可以确保没有引入伸展  
将若干向量的二维坐标投影到一条直线上,从而实现了降维  

To project a matrix A onto a vector v:  
投影矩阵:$A_P=A\frac{v}{||v||_2}=AV$,V是多个列向量组成的矩阵  
投影到两个向量上与投影到这两个向量跨越的平面上是一样的.  
# 52.Motivating PCA(主成分分析动机)  
保留分散意味着保留更多信息  
PCA的目标是找到能够在降低数据维度的同时最大化保留数据分布的投影  
降维使得数据集更易管理,因为它们更小  
PCA允许你在减少维度的同时最小化信息损失  
# 53.Varicance and Covariance(方差和协方差)  
$Var(x)=\frac{1}{n-1}\displaystyle\sum_{i=1}^n(x_i-\mu_x)^2$  
注意:除以n-1而不是n是为了修正偏差,尤其是在小样本统计中  
协方差帮助衡量数据集中两个特征相对于彼此的变化   
$Cov(x,y)=\frac{1}{n-1}\displaystyle\sum_{i=1}^n(x_i-\mu_x)(y_i-\mu_y)$  
一三象限的点对协方差贡献为正,二四象限的点对协方差贡献为负  
将协方差视为衡量两个变量之间关系方向的指标  
负协方差表示负向趋势,小协方差表示平稳趋势或无关系,正协方差表示正向趋势  
# 54.Covirance Matrix(协方差矩阵)  
$Cov(y,x)=Cov(x,y)$  
$Cov(x,x)=Var(x)$  
协方差矩阵:  
$$
C=
\begin{bmatrix}
Var(x) & Cov(x,y)\\
Cov(y,x) & Var(y)
\end{bmatrix}
$$
$$
A=
\begin{bmatrix}
x_1 & y_1\\
x_2 & y_2\\
\vdots &\vdots\\
x_n & y_n
\end{bmatrix}
\mu=
\begin{bmatrix}
\mu_x & \mu_y\\
\mu_x & \mu_y\\
\vdots &\vdots\\
\mu_x &\mu_y
\end{bmatrix}
$$
$C=\frac{1}{n-1}(A-\mu)^T(A-\mu)$  

Matrix formula:  
1.Arrange data with a different feature in each column  
2.Calculate column averages  
3.Substract each average from their respective column to generate $A-\mu$  
4.$\frac{1}{n-1}(A-\mu)^T(A-\mu)$ gives the covariance matrix C  
# 55.PCA Overview(主成分分析概述)  
每个沿其对角线对称的矩阵的特征向量都是正交的  
协方差矩阵的两个特征向量称为principal components(主成分)  
具有最大特征值的特征向量将始终是在投影数据时提供最大方差的那个向量  
# 56.PCA Why it works(主成分分析原理)  
矩阵C的特征向量告诉你可以将矩阵视为仅仅是直接拉伸的方向  
最大的特征值告诉你在哪个方向上拉伸最大,其他任何方向上的拉伸都会较小  
选择具有最大特征值的特征向量将为你提供最大拉伸或最大方差的方向  
# 57.PCA Mathematical Formula(主成分分析数学推导)  
1.Create matrix  
2.Center the data,计算$X-\mu$  
3.Calculate Covariance Matrix,$C=\frac{1}{n-1}(A-\mu)^T(A-\mu)$  
4.Calculate Eigenvectors and Eigenvalues  
将特征值从小到大的顺序对它们进行排序,保留t个最大特征值的特征向量(t为你想降低到的维度)  
5.Create Projection Matrix  
假设t=2,根据保留的两个特征向量进行投影,按照范数进行缩放
$$
V=
\begin{bmatrix}
\frac{v_1}{||v_1||_2} & \frac{v_2}{||v_2||_2}\\
\end{bmatrix}
$$
6.Project Centered Data,最终投影数据   
$_{PCA}=(X-\mu)V$
# 59.Discrete Dynamical Systems(离散动力系统)  
Markov matrix(马尔可夫矩阵):All values are positive and columns add to 1  
它允许你推断系统演化的概率  
State vector(状态向量)  
transition matrix(转移矩阵)  
Equilibrium vector(平衡向量),它也是转移矩阵的特征向量  

































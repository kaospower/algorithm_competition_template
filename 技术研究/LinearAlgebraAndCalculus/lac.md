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





































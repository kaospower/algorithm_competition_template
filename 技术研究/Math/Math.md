# 梯度,散度和旋度  
哈密顿算符:$\nabla$  
$\nabla$表示某一物理量在三个坐标方向的偏导数的矢量和  
$\nabla=\frac{\partial}{\partial x}\mathbf{i}+
\frac{\partial}{\partial y}\mathbf{j}+
\frac{\partial}{\partial z}\mathbf{k}$  

梯度(Gradient)  
标量场的梯度是一个矢量场,它表示s在空间某一位置沿某一方向的变化量  

当$\nabla$作用于标量s时即可得到该标量在空间中的梯度  
$grad~s=\nabla s=
\frac{\partial s}{\partial x}\mathbf{i}+
\frac{\partial s}{\partial y}\mathbf{j}+
\frac{\partial s}{\partial z}\mathbf{k}$


散度(Divergence)  
矢量的散度是一个标量,表示空间中某一区域流入或流出的矢量的多少  

根据矢量点乘的运算规则,$\nabla$与一个矢量的点乘是一个标量,它代表了矢量场的散度  
$div~\mathbf{v}=\nabla\cdot\mathbf{v}=
\frac{\partial u}{\partial x}+
\frac{\partial v}{\partial y}+
\frac{\partial w}{\partial z}$


标量的梯度为矢量,因此对该矢量可以继续求散度,从而引入拉普拉斯算子$\nabla^2$  
$\nabla\cdot(\nabla s)=\nabla^2 s=
\frac{\partial^2s}{\partial x^2}+
\frac{\partial^2s}{\partial y^2}+
\frac{\partial^2s}{\partial z^2}$   
矢量的散度为标量,因此对该标量可以继续求梯度  
$\nabla(\nabla\cdot \mathbf{v})=\nabla^2\mathbf{v}=
(\nabla^2u)\mathbf{i}+(\nabla^2v)\mathbf{j}+(\nabla^2w)\mathbf{k}$  
拉普拉斯算子对标量的运算结果为标量,对矢量的运算结果为矢量  

旋度(Curl)  
旋度是由$\nabla$与矢量叉乘得到,它的运算结果是一个矢量,代表了矢量做旋转运动的方向和强度  
$\nabla\times\mathbf{v}=
(\frac{\partial}{\partial x}\mathbf{i}+
\frac{\partial}{\partial y}\mathbf{j}+
\frac{\partial}{\partial z}\mathbf{k})
\times(u\mathbf{i}+v\mathbf{j}+w\mathbf{k})=
\begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k}\\
\frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z}\\
u & v & w\\
\end{vmatrix}=
(\frac{\partial w}{\partial y}-\frac{\partial v}{\partial z})\mathbf{i}+
(\frac{\partial u}{\partial z}-\frac{\partial w}{\partial x})\mathbf{j}+
(\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y})\mathbf{k}
$  



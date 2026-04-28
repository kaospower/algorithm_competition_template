# 常用Conda命令

```python
在安装了miniconda或anaconda情况下，我们可以使用conda命令
conda env list:列出当前安装的base环境和所有其他环境
conda env remove -n name:移除某个环境(name为该环境名)
conda list:列出当前环境下安装的所有包
conda info:查看当前conda信息
```

# mac安装tensorflow流程

```python
由于mac M1芯片不含有独立的GPU，因此和windows系统安装流程有所不同，因此要特别注意
```

# Install TensorFlow on Macbook M1/M2/M3

[教程链接](https://blog.fotiecodes.com/install-tensorflow-on-your-mac-m1m2m3-with-gpu-support-clqs92bzl000308l8a3i35479)

Step 1:Install Xcode Command Line Tool

Step 2:Install the M1 Miniconda or Anaconda Version

Step 3:Install Tensorflow

Step 4:Install Jupyter Notebook and common packages

Step 5:Check GPU availability



after you install miniconda3

```python
conda create -n tf-gpu python=3.9.6 ## or whatever version of python you want
```

```python
conda activate tf-gpu
```

```python
conda install -c apple tensorflow-deps
```

```python
pip install tensorflow-macos
```

```python
pip install tensorflow-metal
```

```python
conda install notebook -y
```

```python
#当需要安装某个包时，先激活虚拟环境,然后pip install [package] 或者conda install [package]
pip install numpy
pip install pandas
pip install matplotlib 
pip install scikit-learn 
pip install scipy
pip install plotly 
```

# 重要！！！！不要随便升级numpy,pandas等核心包版本，会造成与tensorflow不匹配而导致import tensorflow失败。不要随便用-U命令升级以前的包



```python
#将python包回退到之前某个特定版本
pip install numpy==1.26.4
```

# 启动tensorflow环境

1.打开终端

2.输入conda activate tf-gpu(关闭命令为conda deactivate)

3.输入jupyter notebook

4.import tensorflow as tf-gpu

5.编写代码

### 最新版jupyter notebook不需要代码提示插件了，按TAB自带插件很好用

### tensorflow运行numpy报错:

回退numpy版本到日志显示的二进制版本

显示tensorflow不匹配：回退tensorflow版本

### ERROR: Could not find a [version](https://so.csdn.net/so/search?q=version&spm=1001.2101.3001.7020) that satisfies the requirement trasformers ERROR: No matching distribution found for trasformers

解决办法：

pip install transformers -i https://pypi.python.org/simple

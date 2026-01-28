在官网下载好python后

使用where查看python安装位置

然后vim ~/.zshrc配置环境变量

PYTHONPATH="/Library/Frameworks/Python.framework/Versions/3.13/bin"

export PYTHONPATH

之后source ~/.zhshrc,之后重启电脑(必须)



当前系统已经自动将python解释器位置加入到环境变量中,无需手动添加

在命令行中,如果用python3.13,则输入python3.13回车

在命令行中,如果用python3.9,则输入python3.9回车

如使用当前默认版本的python,输入python3回车



下面命令是用来动态更新默认版本的python

\#设置python3.9

`PYTHON_3_9_PATH=/Library/Frameworks/Python.framework/Versions/3.9/bin`

\#设置python3.13

`PYTHON_3_13_PATH=/Library/Frameworks/Python.framework/Versions/3.13/bin`

`export PATH=$PYTHON_3_13_PATH:$PATH`

\#alias命令动态切换python版本

#这个命令意思是在原来系统环境变量最前面追加上这个变量

`alias python309="export PATH=$PYTHON_3_9_PATH:$PATH"`

`alias python313="export PATH=$PYTHON_3_13_PATH:$PATH"`




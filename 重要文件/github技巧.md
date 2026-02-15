# 将项目添加到git

进入pycharm终端,cd到项目所在目录

初始化git仓库

`git init`

`git add .`

`git commit -m "Initial commit"`

链接到github

`git remote add origin https://github.com/<用户名>/<仓库名>.git`

`git branch -M main`

`git push -u origin main`

# 推送远程仓库失败(貌似是pycharm的bug)
`关闭所有程序,重启电脑`

`git pull origin main --rebase`

`git push origin main`

# 日常推送到远程仓库失败

`git push origin main -v`

或者更换wifi
一种配置代理的解决方案(https://developer.aliyun.com/article/1392370)  

# 从github导入项目

1.新建一个文件夹作为项目目录

2.get from vcs,填入仓库git链接以及当前项目目录,git链接可以点击仓库code按钮获得

# 更换设备同步仓库
先git pull同步远程仓库代码
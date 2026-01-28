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

`git pull origin main --rebase`

`git push origin main`

# 日常推送到远程仓库失败

`git push origin main -v`

或者更换wifi
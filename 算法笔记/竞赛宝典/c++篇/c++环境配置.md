1.在mac中使用<bits/stdc++.h>头文件

在vscode终端使用命令echo | g++ -v -x c++ -E -`中打印出所有include路径

在前三条路径中创建bits文件夹,将stdc++.h文件拷贝进文件夹

注意:mac中xcode只支持部分头文件,<cstdalign>不支持,因此需要把该头文件注释掉

每次xcode更新会重置inlude目录,因此每次更新xcode之后需要重复上述操作
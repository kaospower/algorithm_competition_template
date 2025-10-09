"""
e函数模版
"""
def Z(s):
    n=len(s)
    z=[0]*n
    z[0]=n #z[0]表示s本身
    c=r=1
    #从下标1开始计算
    for i in range(1,n):
        lens=min(r-i,z[i-c]) if r>i else 0
        #涵盖四种情况,情况2和3不执行while,情况1和4执行while
        while i+lens<n and s[i+lens]==s[lens]:
            lens+=1
        if i+lens>r:
            r=i+lens
            c=i
        z[i]=lens
    #返回z数组
    return z

#求解a的e数组
def E(a,b):
    n,m=len(a),len(b)
    z=Z(b) #求解B的Z数组
    e=[0]*n
    c=r=0
    for i in range(n):
        lens=min(r-i,z[i-c]) if r>i else 0
        while i+lens<n and lens<m and a[i+lens]==b[lens]:
            lens+=1
        if i+lens>r:
            r=i+lens
            c=i
        e[i]=lens
    return e
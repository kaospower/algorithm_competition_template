"""
求s,t以i结尾,j开头的最长公共子串长度
"""

def g(s,t):
    n,m=len(s),len(t)
    #f[i+1][j]代表s中以i结尾,t中以j开头的最长匹配长度
    f=[[0]*(m+1) for _ in range(n+1)]
    for i,x in enumerate(s):
        for j,y in enumerate(t):
            if x==y:
                f[i+1][j]=f[i][j+1]+1
    mx=list(map(max,f))
    return mx
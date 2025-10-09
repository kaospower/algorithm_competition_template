"""
求s,t以i,j位置结尾的最长公共子串长度
"""
def g(s,t):
    n,m=len(s),len(t)
    #f[i+1][j+1]表示s,t分别以i,j结尾的最长匹配长度
    f=[[0]*(m+1) for _ in range(n+1)]
    for i,x in enumerate(s):
        for j,y in enumerate(t):
            if s[i]==t[j]:
                f[i+1][j+1]=f[i][j]+1
    mx=list(map(max,f))
    return mx
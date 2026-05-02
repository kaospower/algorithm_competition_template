from functools import cache
"""
数位dp简化模版,只有isLimit参数,这个模版常用于处理可以忽略前导0的问题
"""

#二进制模版
mod=1_000_000_007
def g(n):
    m=n.bit_length()
    #isLimit表示当前是否受到了n的约束
    @cache
    def f(i,isLimit):
        if i==m:
            return 1
        res=0
        up=n>>m-1-i&1 if isLimit else 1 #利用位运算取出n从左往右数第i个二进制位
        for d in range(up+1):
            res+=f(i+1,isLimit and d==up)
        return res%mod
    return f(0,True)


#十进制模版
mod=1_000_000_007
def g(s):
    m=len(s)
    #isLimit表示当前是否受到了n的约束
    @cache
    def f(i,isLimit):
        if i==m:
            return 1
        res=0
        up=int(s[i]) if isLimit else 9 #对于k进制,这里替换成k-1,常用的10进制这里就是9
        for d in range(up+1):
            res+=f(i+1,isLimit and d==up)
        return res%mod
    return f(0,True)
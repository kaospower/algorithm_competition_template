from functools import cache
"""
数位dp标准模版,带isLimit和isNum参数,这个模版常用于处理需要考虑前导0的问题
实际应用时根据题意在f中补充额外的参数
注意此模版不包含0,表示的是从1开始的结果
"""
mod=1_000_000_007
def g(s):
    m=len(s)
    @cache
    #isLimit表示当前是否受到了n的约束,isNum 表示i前面的数位是否填了数字
    def f(i,isLimit,isNum):
        if i==m:
            return 1 if isNum else 0
        res=0
        #跳过当前数位
        if not isNum:
            res=f(i+1,False,False)
        #不跳过
        low=0 if isNum else 1
        up=int(s[i]) if isLimit else 9 #对于k进制,这里替换成k-1,常用的10进制这里就是9
        for d in range(low,up+1):
            res+=f(i+1,isLimit and d==up,True)
        return res%mod
    return f(0,True,False)
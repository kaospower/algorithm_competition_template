from functools import cache
"""
上下界数位dp精简模版,带isLimit和isNum参数,这个模版常用于处理不需要考虑前导0的问题
"""
mod=1_000_000_007
def g(num1,num2):
    n=len(num2)
    num1=num1.zfill(n) #补前导0
    @cache
    def f(i,limit_low,limit_high):
        if i==n:
            return 1

        lo=int(num1[i]) if limit_low else 0
        hi=int(num2[i]) if limit_high else 9 #k进制时换成k-1
        res=0
        for d in range(lo,hi+1):
            res+=f(i+1,limit_low and d==lo,limit_high and d==hi)
        return res%mod
    return f(0,True,True)
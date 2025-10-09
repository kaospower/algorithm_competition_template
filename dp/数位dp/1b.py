from functools import cache
"""
数位dp简化模版,只有isLimit参数,这个模版常用于处理可以忽略前导0的问题
由于可以忽略前导0的影响,我们可以用位运算优化部分操作
注意此模版不包含0,表示的是从1开始的结果
"""
mod=1_000_000_007
def g(n):
    m=n.bit_length()
    #isLimit表示当前是否受到了n的约束,isNum 表示i前面的数位是否填了数字
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
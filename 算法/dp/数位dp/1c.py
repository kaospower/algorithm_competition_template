from functools import cache
"""
上下界数位dp标准模版,带isLimit和isNum参数,这个模版常用于处理需要考虑前导0的问题
"""
mod=1_000_000_007
#num1,num2分别代表下界和上界
def h(num1,num2):
    def g(num):
        m=len(num)
        #isLimit 表示当前是否受到了 n 的约束,isNum 表示 i 前面的数位是否填了数字
        @cache
        def f(i,isLimit,isNum):
            if i==m:
                return 1 if isNum else 0
            res=0
            #跳过当前数位
            if not isNum:
                res=f(i+1,False,False)
            #不跳过
            low=0 if isNum else 1
            up=int(num[i]) if isLimit else 9
            for d in range(low,up+1):
                #注意此时只有d等于上界并且isLimit原来为True,新的isLimit才是True
                res+=f(i+1,isLimit and d==up,True)
            return res%mod
        return f(0,True,False)
    return (g(num2)-g(str(int(num1)-1)))%mod
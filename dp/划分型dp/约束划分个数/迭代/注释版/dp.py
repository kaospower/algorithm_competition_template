from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

#约束划分个数为K(注意是大写),迭代模版
#模版题:3599(https://leetcode.cn/problems/partition-array-to-minimize-xor/description/)
fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y
def solve(nums,K):
    n=len(nums)
    #k的范围为0~K,长度K+1
    #i的范围是0~n-1,但是为了避免边界讨论,用f[i+1]表示以i结尾的结果,即实际下标比真正下标偏移1
    #因此i的范围是0~n,长度n+1
    #由于k<=n,k的范围更小,因此写在外层
    #默认边界值是inf,代表不合法
    f=[[inf]*(n+1) for _ in range(K+1)]
    #f[0][0]对应记忆化搜索中的边界,即k=0,i=-1时的情况
    f[0][0]=0
    #k从1开始,因为k=0时,除了i=0,其他都不合法,为inf,i=0的情况已经特判完成
    for k in range(1,K+1):
        #i代表实际下标偏移1,i用+1下标是因为dp数组用的是偏移后的下标,因此提前在循环中加上偏移的1
        #左边要留出k-1个位置,因此左边界为k-1,右边要留出已经划分完的K-k个位置,因此右边界为n-(K-k)-1
        #由于python循环上界是开区间,因此右边界是n-(K-k)
        #由于i实际下标比真正下标要偏移1,因此这里直接在循环中加上1,因此上下界变成k,n-(K-k)+1
        #所以循环中写成range(k,n-(K-k)+1)
        for i in range(k,n-(K-k)+1):
            s=0 #s从右往左算数组异或和
            ans=inf #ans记录最值
            #j代表实际下标,即不加1,因为i加了1,因此j的上界为i-1,即将偏移量减去
            #j用实际下标是因为计算数组异或和要用实际下标,因此提前在循环中减掉偏移的1
            #j左边必须留出k-1个位置给剩下k-1个划分,下标为0~k-2
            #因此剩下划分的结尾位置为k-2,而j的最左下标刚好是k-2+1=k-1
            #由于python循环是左闭右开区间,因此范围写成range(i-1,k-2,-1)
            for j in range(i-1,k-2,-1):
                s^=nums[j]
                #j代表当前划分的开头,j左边即是下一个划分的结尾位置,即要转移到的地方,同时k-1
                ans=fmin(ans,fmax(s,f[k-1][j]))
            #注意i在上面已经加完1了
            f[k][i]=ans
    return f[-1][-1]




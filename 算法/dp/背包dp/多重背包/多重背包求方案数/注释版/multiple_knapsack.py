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

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y

#多重背包求方案数模版
#利用离散前缀和优化
#背包问题求指定体积V的方案数,有两种写法
#一种是f=[1]+[0]*V,这样f[j]表示恰好装满j的方案数
#另一种是f=[1]*(V+1),这样f[j]表示至多装j的方案数
#为了避免出现错误,建议使用第一种写法
#模版题:2902(https://leetcode.cn/problems/count-of-sub-multisets-with-bounded-sum/description/)
fmin=lambda x,y:x if x<y else y
mod=1_000_000_007
#arr_v代表物品体积数组,arr_c代表物品数量数组,V代表背包容量,C代表体积为0的物品的数量,l,r代表体积上下界
def solve(arr_v,arr_c,V,C,l,r):
    f=[1+C]+[0]*V
    s=0
    for v,c in zip(V,C):
        #s用来动态更新体积枚举的上界,当前物品最多选c个,之前体积上界是s,全选当前物品,上界为s+v*c
        #同时s不能超过r
        s=fmin(s+v*c,V)
        #在f数组上原地更新前缀和
        #注意本题物品有体积v,因此前缀和不是连续的,是类似f[j]+f[j-v]+f[j-2*v]...的离散前缀和
        for j in range(v,s+1):
            f[j]=(f[j]+f[j-v])%mod
        #注意本题相当于带权值的前缀和
        #f[j]表示j,j-v,j-2*v,....的和
        #如果求j,j-v,...j-c*v的和,需要用f[j]-f[j-(c+1)*v]
        #这和普通前缀和f[i...j]=s[i]-s[j-1]是一样的道理,由于本题是权值前缀和,因此不能直接-1,要额外-v
        t=(c+1)*v
        for j in range(s,t-1,-1):
            f[j]=(f[j]-f[j-t])%mod
    return sum(f[l:])%mod

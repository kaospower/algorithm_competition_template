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
fmin=lambda x,y:x if x<y else y
mod=1_000_000_007
#arr_v代表物品体积数组,arr_c代表物品数量数组,V代表背包容量,C代表体积为0的物品的数量,l,r代表体积上下界
def solve(arr_v,arr_c,V,C,l,r):
    f=[1+C]+[0]*V
    s=0
    for v,c in zip(arr_v,arr_c):
        s=fmin(s+v*c,V)
        for j in range(v,s+1):
            f[j]=(f[j]+f[j-v])%mod
        t=(c+1)*v
        for j in range(s,t-1,-1):
            f[j]=(f[j]-f[j-t])%mod
    return sum(f[l:])%mod

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
fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y
def solve(nums,K):
    n=len(nums)
    f=[[inf]*(n+1) for _ in range(K+1)]
    f[0][0]=0
    for k in range(1,K+1):
        for i in range(k,n-(K-k)+1):
            s=0
            ans=inf
            for j in range(i-1,k-2,-1):
                s^=nums[j]
                ans=fmin(ans,fmax(s,f[k-1][j]))
            f[k][i]=ans
    return f[-1][-1]




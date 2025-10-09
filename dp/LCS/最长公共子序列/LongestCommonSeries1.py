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

#最长公共子序列(二维数组版本)
def g(s,t):
    n,m=len(s),len(t)
    if m>n:
        return g(t,s)
    f=[[0]*(m+1) for _ in range(n+1)]
    for i,x in enumerate(s):
        for j,y in enumerate(t):
            f[i+1][j+1]=f[i][j]+1 if x==y else fmax(f[i+1][j],f[i][j+1])
    return f[-1][-1]

#模版题见L1143(https://leetcode.cn/problems/longest-common-subsequence/description/)

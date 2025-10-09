from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd,isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

fmax=lambda x,y:x if x>y else y
fmin=lambda x,y:x if x<y else y

#最长公共子序列(空间压缩版本)
def g(s,t):
    n,m=len(s),len(t)
    if m>n:
        return g(t,s)
    f=[0]*(m+1)
    for i,x in enumerate(s):
        tmp=0
        for j,y in enumerate(t):
            pre,tmp=tmp,f[j+1]
            f[j+1]=pre+1 if x==y else fmax(f[j],f[j+1])
    return f[-1]

#模版题见L1143(https://leetcode.cn/problems/longest-common-subsequence/description/)

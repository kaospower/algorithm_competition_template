from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise,chain

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

#leetcode 3742(https://leetcode.cn/problems/maximum-path-score-in-a-grid/description/)
#网格图dp
class Solution:
    def maxPathScore(self, grid: List[List[int]], k: int) -> float | int:
        n,m=len(grid),len(grid[0])
        @cache
        def f(i,j,k):
            if i<0 or j<0 or k<0:
                return -inf
            if i==0 and j==0:
                return 0
            if grid[i][j]>0:
                k-=1
            return fmax(f(i,j-1,k),f(i-1,j,k))+grid[i][j]
        ans=f(n-1,m-1,k)
        f.cache_clear()
        return ans if ans>=0 else -1


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

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

#leetcode3070(https://leetcode.cn/problems/count-submatrices-with-top-left-element-and-sum-less-than-k/)
#二维前缀和
class Solution:
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        n,m=len(grid),len(grid[0])
        s=[[0]*(m+1) for _ in range(n+1)]
        ans=0
        for i,row in enumerate(grid):
            for j,x in enumerate(row):
                s[i+1][j+1]=s[i+1][j]+s[i][j+1]-s[i][j]+x
                if s[i+1][j+1]<=k:
                    ans+=1
        return ans
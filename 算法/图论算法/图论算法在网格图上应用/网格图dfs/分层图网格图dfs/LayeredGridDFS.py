from typing import List, Optional
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

# leetcode 3882(https://leetcode.cn/problems/minimum-xor-path-in-a-grid/)
#分层图网格图dfs,分层图,即将网格图的点和不同状态的组合看成新的点
#由于是dfs,因此可能会多次访问重复的点,因此需要用vis标记重复点
#同时加上最优性剪枝,最小异或值是0,如果已经找到答案,剩下的路径都没必要走了
class Solution:
    def minCost(self, grid: list[list[int]]) -> int:
        n,m=len(grid),len(grid[0])
        ans=inf
        vis=set()
        def f(i,j,v):
            nonlocal ans
            if ans==0 or i<0 or j<0 or (i,j,v) in vis:
                return
            vis.add((i,j,v))
            v^=grid[i][j]
            if i==0 and j==0:
                ans=min(ans,v)
                return
            f(i-1,j,v)
            f(i,j-1,v)
        f(n-1,m-1,0)
        return ans
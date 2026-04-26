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

# leetcode 1559(https://leetcode.cn/problems/detect-cycles-in-2d-grid/description/)
# 本质仍是无向图找环
# dfs解法,记录上一步位置,类似无向图找环
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        n,m=len(grid),len(grid[0])
        def f(x,y,px,py):
            vis.add((x,y))
            for dx,dy in [(0,1),(0,-1),(-1,0),(1,0)]:
                nx,ny=x+dx,y+dy
                if 0<=nx<n and 0<=ny<m and grid[nx][ny]==grid[x][y] and not (nx==px and ny==py) and ((nx,ny) in vis or f(nx,ny,x,y)):
                    return True
            return False
        vis=set()
        for i,row in enumerate(grid):
            for j,v in enumerate(row):
                if (i,j) not in vis and f(i,j,-1,-1):
                    return True
        return False
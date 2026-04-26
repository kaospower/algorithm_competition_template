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
# 并查集解法,略微修改一下union函数
# 将相邻两个格子看成一条边,为避免重复遍历,对于每个点,向其右侧和下方连一条边
class UnionFind:
    def __init__(self, n):
        self.father = list(range(n))
        self.cc = n

    def find(self, i):
        if i != self.father[i]:
            self.father[i] = self.find(self.father[i])
        return self.father[i]

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx == fy:
            return True
        self.father[fx] = fy
        self.cc -= 1
        return False

    def reset(self):
        n = len(self.father)
        self.father = list(range(n))
        self.cc = n

    def separate(self, x, y):
        self.father[x], self.father[y] = x, y
        self.cc += 1
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        n,m=len(grid),len(grid[0])
        uf=UnionFind(n*m)
        for i,row in enumerate(grid):
            for j,x in enumerate(row):
                if j<m-1 and x==grid[i][j+1] and uf.union(i*m+j,i*m+j+1):
                    return True
                if i<n-1 and x==grid[i+1][j] and uf.union(i*m+j,(i+1)*m+j):
                    return True
        return False

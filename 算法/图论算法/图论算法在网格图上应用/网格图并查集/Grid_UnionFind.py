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


#网格图并查集
#模版题:1631(https://leetcode.cn/problems/path-with-minimum-effort/description/)
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        n,m=len(heights),len(heights[0])
        tot=n*m
        u=UnionFind(tot)
        def f(x,y):
            return x*m+y
        edges=[]
        for i in range(n):
            for j in range(m):
                for dx,dy in [(1,0),(-1,0),(0,-1),(0,1)]:
                    nx,ny=i+dx,j+dy
                    if 0<=nx<n and 0<=ny<m:
                        edges.append((abs(heights[nx][ny]-heights[i][j]),f(i,j),f(nx,ny)))
        edges.sort()
        ans=0
        for w,x,y in edges:
            u.union(x,y)
            ans=max(ans,w)
            if u.find(0)==u.find(tot-1):
                return ans
        return ans

# 带标签和返回值的并查集,存储形式为数组
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
            return False
        self.father[fx] = fy
        self.cc -= 1
        return True

    def reset(self):
        n = len(self.father)
        self.father = list(range(n))
        self.cc = n

    def separate(self, x, y):
        self.father[x], self.father[y] = x, y
        self.cc += 1

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


#网格图并查集将相邻格子之间看成有一条边,然后遍历每个格子,考虑它和右边邻居以及下面邻居是否可以合并
#实际实现时通常将二维坐标通过i*m+j映射成一维,方便处理

#leetcode 1391(https://leetcode.cn/problems/check-if-there-is-a-valid-path-in-a-grid/description/)
#考虑相邻两个格子,如果一个向右的开口另一个有向左的开口,则可以横向连通
#如果一个有向下另一个有向上的开口,则可以纵向连通
#并查集合并时根据这一规则即可,最后检验左上角和右下角是否连通
#实现时,记录上下左右开口各自包含哪些种类的格子即可
#然后对于水平相邻格子检验横向连通,对于垂直相邻格子检验纵向连通

class Solution:
    def hasValidPath(self, grid: List[List[int]]) -> bool:
        #1:LR,2:UD,3:LD,4:RD,5:LU,6:RU
        d={'L':[1,3,5],'R':[1,4,6],'U':[2,5,6],'D':[2,3,4]}
        n,m=len(grid),len(grid[0])
        uf=UnionFind(n*m)
        for i,row in enumerate(grid):
            for j,x in enumerate(row):
                if j<m-1 and x in d['R'] and grid[i][j+1] in d['L']:
                    uf.union(i*m+j,i*m+j+1)
                if i<n-1 and x in d['D'] and grid[i+1][j] in d['U']:
                    uf.union(i*m+j,(i+1)*m+j)
        return uf.union(0,m*n-1)



class UnionFind:
    def __init__(self,n):
        self.father=list(range(n))
        self.cc=n
    def find(self,i):
        if i!=self.father[i]:
            self.father[i]=self.find(self.father[i])
        return self.father[i]
    def union(self,x,y):
        fx,fy=self.find(x),self.find(y)
        if fx==fy:
            return True
        self.father[fx]=fy
        self.cc-=1
        return False
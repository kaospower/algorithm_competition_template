from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache, reduce
from math import gcd, isqrt
from operator import and_, or_, xor, add, mul
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

# leetcode 1914(https://leetcode.cn/problems/cyclically-rotating-a-grid/description/)
# 利用坐标向量模拟螺旋矩阵
DIRS=(0,1),(1,0),(0,-1),(-1,0) #右下左上顺时针遍历
class Solution:
    def rotateGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        n,m=len(grid),len(grid[0])
        for i in range(min(n,m)//2):
            n0,m0=n-i*2,m-i*2
            x,y=i,i
            a=[]
            for dx,dy in DIRS:
                for _ in range(m0-1):
                    a.append(grid[x][y])
                    x+=dx
                    y+=dy
                n0,m0=m0,n0 #交换行和列维度,因为长和宽需要遍历元素数目不等
            shift=k%len(a)
            a=a[shift:]+a[:shift] #逆时针轮转shift位

            j=0
            for dx,dy in DIRS:
                for _ in range(m0-1):
                    grid[x][y]=a[j]
                    j+=1
                    x+=dx
                    y+=dy
                n0,m0=m0,n0
        return grid
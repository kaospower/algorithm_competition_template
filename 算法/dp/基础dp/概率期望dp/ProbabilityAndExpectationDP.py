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

# leetcode 688(https://leetcode.cn/problems/knight-probability-in-chessboard/

#概率期望dp
DIRS=(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(-1,-2),(1,-2),(-1,2)
class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        #f(i,j,k)表示马从(i,j)出发,走k步后依然在棋盘上的概率
        #马踏八方,走每个方向的概率均为1/8
        @cache
        def f(i,j,k):
            if i<0 or i>=n or j<0 or j>=n:
                return 0
            if k==0:
                return 1
            return sum(f(i+dx,j+dy,k-1)for dx,dy in DIRS)/8
        return f(row,column,k)

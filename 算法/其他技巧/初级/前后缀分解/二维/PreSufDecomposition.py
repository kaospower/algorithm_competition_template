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

# leetcode2906(https://leetcode.cn/problems/construct-product-matrix/)
# 前后缀分解
mod = 12345
class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        p = [[0] * m for _ in range(n)]

        # 后缀乘积
        t = 1
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                p[i][j] = t
                t = t * grid[i][j] % mod

        # 计算前缀乘积同时更新答案
        t = 1
        for i in range(n):
            for j in range(m):
                p[i][j] = p[i][j] * t % mod
                t = t * grid[i][j] % mod
        return p

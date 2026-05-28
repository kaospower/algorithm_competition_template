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

#leetcode(https://leetcode.cn/problems/maximum-non-negative-product-in-a-matrix/description/)
# 网格图dp+归纳+边界讨论
class Solution:
    def maxProductPath(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])

        @cache
        def f(i, j):
            x = grid[i][j]
            if i == 0 and j == 0:
                return x, x
            rmx, rmn = -inf, inf
            if i > 0:
                mx, mn = f(i - 1, j)
                rmx = max(mx * x, mn * x)
                rmn = min(mn * x, mx * x)
            if j > 0:
                mx, mn = f(i, j - 1)
                rmx = max(rmx, mx * x, mn * x)
                rmn = min(rmn, mx * x, mn * x)
            return rmx, rmn

        mx, mn = f(n - 1, m - 1)
        return -1 if mx < 0 else mx % 1_000_000_007

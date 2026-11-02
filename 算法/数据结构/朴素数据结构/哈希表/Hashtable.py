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

# leetcode 1001(https://leetcode.cn/problems/grid-illumination/)
# 五个哈希表
class Solution:
    def gridIllumination(self, n: int, lamps: List[List[int]], queries: List[List[int]]) -> List[int]:
        row = defaultdict(int)
        col = defaultdict(int)
        dl = defaultdict(int)
        dr = defaultdict(int)
        d = set()
        size = len(queries)
        ans = [0] * size
        for x, y in lamps:
            if (x, y) in d:
                continue
            row[x] += 1
            col[y] += 1
            dl[x - y] += 1
            dr[x + y] += 1
            d.add((x, y))
        for i, (x, y) in enumerate(queries):
            if row[x] or col[y] or dl[x - y] or dr[x + y]:
                ans[i] = 1
            for nx in range(x - 1, x + 2):
                for ny in range(y - 1, y + 2):
                    if 0 <= nx < n and 0 <= ny < n and (nx, ny) in d:
                        d.remove((nx, ny))
                        row[nx] -= 1
                        col[ny] -= 1
                        dl[nx - ny] -= 1
                        dr[nx + ny] -= 1
        return ans
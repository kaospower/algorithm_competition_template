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

#模拟+对角线前缀和
#leetcode1878(https://leetcode.cn/problems/get-biggest-three-rhombus-sums-in-a-grid/description/)
class Solution:
    def getBiggestThree(self, grid: List[List[int]]) -> List[int]:
        m, n = len(grid), len(grid[0])
        diag_sum = [[0] * (n + 1) for _ in range(m + 1)]  # ↘ 前缀和
        anti_sum = [[0] * (n + 1) for _ in range(m + 1)]  # ↙ 前缀和
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                diag_sum[i + 1][j + 1] = diag_sum[i][j] + v
                anti_sum[i + 1][j] = anti_sum[i][j + 1] + v

        # 从 (x,y) 开始，向 ↘，连续 k 个数的和
        def query_diagonal(x: int, y: int, k: int) -> int:
            return diag_sum[x + k][y + k] - diag_sum[x][y]

        # 从 (x,y) 开始，向 ↙，连续 k 个数的和
        def query_anti_diagonal(x: int, y: int, k: int) -> int:
            return anti_sum[x + k][y + 1 - k] - anti_sum[x][y + 1]

        x = y = z = 0  # 最大，次大，第三大

        def update(v: int) -> None:
            nonlocal x, y, z
            if v > x:
                x, y, z = v, x, y
            elif x > v > y:
                y, z = v, y
            elif y > v > z:
                z = v

        # 枚举菱形正中心 (i,j)
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                update(v)  # 一个数也算菱形
                # 枚举菱形顶点到正中心的距离 k，注意菱形顶点不能出界
                # i-k >= 0 且 i+k <= m-1，所以 k <= min(i, m-1-i)，对于 j 同理
                mx = min(i, m - 1 - i, j, n - 1 - j)
                for k in range(1, mx + 1):
                    a = query_diagonal(i - k, j, k)                   # 菱形右上的边
                    b = query_diagonal(i, j - k, k)                   # 菱形左下的边
                    c = query_anti_diagonal(i - k + 1, j - 1, k - 1)  # 菱形左上的边
                    d = query_anti_diagonal(i, j + k, k + 1)          # 菱形右下的边
                    update(a + b + c + d)

        ans = [x, y, z]
        while ans[-1] == 0:  # 不同的和少于三个
            ans.pop()
        return ans


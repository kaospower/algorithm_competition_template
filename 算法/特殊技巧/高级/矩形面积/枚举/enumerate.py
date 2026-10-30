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

#leetcode1727(https://leetcode.cn/problems/largest-submatrix-with-rearrangements/)
#矩形+枚举
class Solution:
    def largestSubmatrix(self, matrix: List[List[int]]) -> int:
        n = len(matrix[0])
        heights = [0] * n
        ans = 0

        for row in matrix:  # 枚举子矩形的底边
            for j, x in enumerate(row):
                if x == 0:
                    heights[j] = 0
                else:
                    heights[j] += 1

            hs = sorted(heights)  # 复制一份 heights 再排序
            for i, h in enumerate(hs):  # 把 hs[i:] 作为子数组
                # 子数组长为 n-i，最小值为 h，对应的子矩形面积为 (n-i)*h
                ans = max(ans, (n - i) * h)

        return ans
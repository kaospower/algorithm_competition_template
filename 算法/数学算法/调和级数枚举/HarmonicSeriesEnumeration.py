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

# leetcode 3447(https://leetcode.cn/problems/assign-elements-to-groups-with-constraints/)
# 调和级数枚举，思路衍生于埃氏筛
class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        mx = max(groups)
        target = [-1] * (mx + 1)
        for i, x in enumerate(elements):
            if x > mx or target[x] >= 0:
                continue
            for y in range(x, mx + 1, x):
                if target[y] < 0:
                    target[y] = i
        return [target[x] for x in groups]


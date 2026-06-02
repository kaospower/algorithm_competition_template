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

# leetcode 2126(https://leetcode.cn/problems/destroying-asteroids/description/)
# 贪心+二进制长度分组
class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        max_width = max(asteroids).bit_length()
        mn = [inf] * max_width
        sum_ = [0] * max_width

        for x in asteroids:
            i = x.bit_length() - 1
            mn[i] = min(mn[i], x)
            sum_[i] += x

        for m, s in zip(mn, sum_):
            if m == inf:
                continue
            if mass < m:
                return False
            mass += s
        return True

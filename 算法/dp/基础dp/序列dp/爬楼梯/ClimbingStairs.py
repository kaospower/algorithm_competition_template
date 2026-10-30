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


# 爬楼梯
class Solution:
    def climbStairs(self, n: int) -> int:
        @cache
        def f(i):
            if i <= 1:
                return 1
            return f(i - 1) + f(i - 2)

        return f(n)


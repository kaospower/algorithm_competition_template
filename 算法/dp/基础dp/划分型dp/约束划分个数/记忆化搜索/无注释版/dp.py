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

#约束划分个数为k,记忆化搜索模版
fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y
def solve(nums, k):
    @cache
    def f(i, k):
        if k == 0:
            return 0 if i == -1 else inf
        s = 0
        ans = inf
        for j in range(i, k - 2, -1):
            s ^= nums[j]
            ans = fmin(ans, fmax(s, f(j - 1, k - 1)))
        return ans

    ans = f(len(nums) - 1, k)
    f.cache_clear()
    return ans

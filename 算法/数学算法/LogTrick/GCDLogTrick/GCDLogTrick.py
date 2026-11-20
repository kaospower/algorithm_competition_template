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


#GCD LogTrick
class Solution:
    def subarrayGCD(self, nums: List[int], k: int) -> int:
        ans = 0
        a = []  # [GCD，相同 GCD 区间的右端点]
        i0 = -1
        for i, x in enumerate(nums):
            if x % k:  # 保证后续求的 GCD 都是 k 的倍数
                a = []
                i0 = i
                continue
            a.append([x, i])
            # 原地去重，因为相同的 GCD 都相邻在一起
            j = 0
            for p in a:
                p[0] = gcd(p[0], x)
                if a[j][0] != p[0]:
                    j += 1
                    a[j] = p
                else:
                    a[j][1] = p[1]
            del a[j + 1:]
            if a[0][0] == k:  # a[0][0] >= k
                ans += a[0][1] - i0
        return ans     
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

#leetcode2078(https://leetcode.cn/problems/two-furthest-houses-with-different-colors/description)
class Solution:
    def maxDistance(self, colors: List[int]) -> int:
        n = len(colors)
        ans = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[i] != colors[j]:
                    ans = max(ans, j - i)
        return ans

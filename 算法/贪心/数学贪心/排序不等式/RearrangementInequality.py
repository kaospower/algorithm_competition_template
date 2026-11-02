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

#leetcode 1402(https://leetcode.cn/problems/reducing-dishes/description/)
#倒序排序+贪心
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        satisfaction.sort(reverse=True)
        f=0
        for x in accumulate(satisfaction):
            if x<=0:
                return f
            f+=x
        return f
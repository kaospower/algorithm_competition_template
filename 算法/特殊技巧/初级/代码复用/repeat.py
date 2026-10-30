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

# leetcode 3635(https://leetcode.cn/problems/earliest-finish-time-for-land-and-water-rides-ii/description/)
# 贪心+代码复用
class Solution:
    def solve(self, landStartTime: List[int], landDuration: List[int], waterStartTime: List[int], waterDuration: List[int]) -> int:
        t=min(x+y for x,y in zip(landStartTime,landDuration))
        return min(max(t,x)+y for x,y in zip(waterStartTime,waterDuration))

    def earliestFinishTime(self, landStartTime: List[int], landDuration: List[int], waterStartTime: List[int], waterDuration: List[int]) -> int:
        return min(self.solve(landStartTime,landDuration,waterStartTime,waterDuration),
                   self.solve(waterStartTime,waterDuration,landStartTime,landDuration))

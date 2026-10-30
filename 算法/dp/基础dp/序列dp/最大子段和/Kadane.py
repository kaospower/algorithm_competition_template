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

# leetcode 53(https://leetcode.cn/problems/maximum-subarray/)
#最大子段和(Kadane算法)
class Solution:
    def maxSubArray(self, nums: list[int]) -> int:
        #以i结尾的最大子段和
        @cache
        def f(i):
            if i==0:
                return nums[0]
            #如果f(i-1)为负,就不要f(i-1),只保留nums[i]
            return max(f(i-1),0)+nums[i]
        return max(f(i) for i in range(len(nums)))

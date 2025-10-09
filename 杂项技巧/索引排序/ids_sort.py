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

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

#索引排序
def f(nums):
    ids=sorted(range(len(nums)),key=lambda p:nums[p])
    return ids


#模版题2099(https://leetcode.cn/problems/find-subsequence-of-length-k-with-the-largest-sum/description/)
class Solution:
    def maxSubsequence(self, nums: List[int], k: int) -> List[int]:
        ids=sorted(range(len(nums)),key=lambda p:nums[p])
        return [nums[i] for i in sorted(ids[-k:])]
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

# leetcode 154(https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/)
# 对于区间左右端点数字相等的情况,去除右端点向下递归即可
# 开区间写法
class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = -1, n - 1
        while l + 1 < r:
            mid = (l + r) >> 1
            if nums[mid] == nums[r]:
                r -= 1
            elif nums[mid] < nums[r]:
                r = mid
            else:
                l = mid
        return nums[r]

# #闭区间写法
# class Solution:
#     def findMin(self, nums: List[int]) -> int:
#         n=len(nums)
#         l,r=-1,n-2
#         while l<=r:
#             mid=(l+r)>>1
#             if nums[mid]==nums[r+1]:
#                 r-=1
#             elif nums[mid]<nums[r+1]:
#                 r=mid-1
#             else:
#                 l=mid+1
#         return nums[l]


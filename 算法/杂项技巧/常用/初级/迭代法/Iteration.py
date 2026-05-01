from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise,chain

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

# leetcode 396(https://leetcode.cn/problems/rotate-function/description/)
#迭代法,即观察规律进行迭代操作
class Solution:
    def maxRotateFunction(self, nums: List[int]) -> int:
        ans=f=sum(i*x for i,x in enumerate(nums))
        n=len(nums)
        s=sum(nums)
        # f=(f-(n-1)*nums[i])+s-nums[i]
        # f=f+s-nums[i]-(n-1)*nums[i]
        # f=f+s-n*nums[i]
        # f+=s-n*nums[i]
        for i in range(n-1,0,-1):
            f+=s-n*nums[i]
            ans=fmax(ans,f)
        return ans

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


#互质+数学推导
class Solution:
    def maxLength(self, nums: List[int]) -> int:
        ans=2
        mul=1
        l=0
        for r,x in enumerate(nums):
            while gcd(mul,x)>1:
                mul//=nums[l]
                l+=1
            mul*=x
            ans=max(ans,r-l+1)
        return ans
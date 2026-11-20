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


#值域dp
class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        f=defaultdict(int)
        #f[x]表示子序列的最后一个数是x时,子序列的最大长度
        for x in nums:
            #将x加1
            f[x+1]=f[x]+1
            #不变
            f[x]=f[x-1]+1
        return max(f.values())

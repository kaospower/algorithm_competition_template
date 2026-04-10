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

#式子变形
#leetcode 3741(https://leetcode.cn/problems/minimum-distance-between-three-equal-elements-ii/description/)
#将原式子化简成2(k-i)
class Solution:
    def minimumDistance(self, nums: List[int]) -> int:
        d=defaultdict(list)
        ans=inf
        for i,x in enumerate(nums):
            if len(d[x])<2:
                d[x].append(i)
            else:
                ans=min(ans,2*(i-d[x][0]))
                d[x][0],d[x][1]=d[x][1],i
        return -1 if ans==inf else ans
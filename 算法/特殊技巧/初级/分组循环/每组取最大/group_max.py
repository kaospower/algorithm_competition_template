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

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y


#分组循环,每组取最大
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        ans=mx=0
        for i,x in enumerate(neededTime):
            ans+=x
            mx=fmax(mx,x)
            if i==len(neededTime)-1 or colors[i]!=colors[i+1]:
                ans-=mx
                mx=0
        return ans

#模版题:1578(https://leetcode.cn/problems/minimum-time-to-make-rope-colorful/description/)
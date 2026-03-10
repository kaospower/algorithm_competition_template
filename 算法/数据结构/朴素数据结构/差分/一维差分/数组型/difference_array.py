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

#朴素数组实现
#模版题:1094(https://leetcode.cn/problems/car-pooling/)
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        d=[0]*1001
        for v,l,r in trips:
            d[l]+=v
            d[r]-=v
        return all(x<=capacity for x in accumulate(d))
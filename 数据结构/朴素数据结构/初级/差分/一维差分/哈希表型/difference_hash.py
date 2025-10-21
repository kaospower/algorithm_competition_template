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

#利用哈希表+排序模拟C++中的平衡树(map)
#模版题:1094(https://leetcode.cn/problems/car-pooling/)
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        d=defaultdict(int)
        for v,l,r in trips:
            d[l]+=v
            d[r]-=v
        s=0
        for k in sorted(d):
            s+=d[k]
            if s>capacity:
                return False
        return True
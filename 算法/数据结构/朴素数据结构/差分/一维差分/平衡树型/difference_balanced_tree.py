from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

from sortedcontainers import SortedList,SortedDict
# from itertools import pairwise

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y

#利用有序哈希表模拟C++中的平衡树(map)
#模版题:1094(https://leetcode.cn/problems/car-pooling/)
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        d=SortedDict()
        for v,l,r in trips:
            d[l]=d.get(l,0)+v
            d[r]=d.get(r,0)-v
        s=0
        for v in d.values():
            s+=v
            if s>capacity:
                return False
        return True
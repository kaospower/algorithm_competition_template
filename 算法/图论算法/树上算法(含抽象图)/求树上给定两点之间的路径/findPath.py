from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd
from collections import defaultdict, deque

# from sortedcontainers import SortedList
# from itertools import pairwise

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

# 求给定两点start,end之间的路径
def query(start, end, g):
    path = []

    def f(u, fa):
        if u == end:
            path.append(u)
            return True
        for v in g[u]:
            if v == fa:
                continue
            if f(v, u):
                path.append(u)
                return True
        return False

    f(start, -1)
    return path

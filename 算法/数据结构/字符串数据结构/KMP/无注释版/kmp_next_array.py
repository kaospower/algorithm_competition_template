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

"""
单纯求解next数组
"""
def ne(p):
    m = len(p)
    ne = [0] * m

    j = 0
    for i in range(1, m):
        while j and p[j] != p[i]:
            j = ne[j - 1]
        if p[j] == p[i]:
            j += 1
        ne[i] = j
    return ne
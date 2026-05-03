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
kmp模版1
s代表文本串,p代表模式串
返回模式串p在文本串中的第一个匹配位置,如果不存在匹配位置返回-1
时间复杂度O(n+m)
"""
def kmp(s, p):
    n, m = len(s), len(p)
    ne = [0] * m
    j = 0
    for i in range(1, m):
        while j and p[j] != p[i]:
            j = ne[j - 1]
        if p[j] == p[i]:
            j += 1
        ne[i] = j

    j = 0
    for i, v in enumerate(s):
        while j and p[j] != v:
            j = ne[j - 1]
        if p[j] == v:
            j += 1
        if j == m:
            return i - m + 1
    return -1
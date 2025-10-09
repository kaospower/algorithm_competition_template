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

#最小质因子lpf
N=1_000_000
lpf=[0]*(N+1)
for i in range(2,N+1):
    if lpf[i]==0:
        for j in range(i,N+1,i):
            if lpf[j]==0:
                lpf[j]=i

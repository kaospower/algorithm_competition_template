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

#预处理因子
N=100_000
p=[[] for _ in range(N+1)]
for i in range(1,N+1):
    for j in range(i,N+1,i):
        p[j].append(i)

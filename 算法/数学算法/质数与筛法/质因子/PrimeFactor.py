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

#预处理每个数的质因子列表,思路类似埃氏筛
N=100_000
p=[[] for _ in range(N+1)]
for i in range(2,N+1):
    if not p[i]:
        for j in range(i,N+1,i):
            p[j].append(i)

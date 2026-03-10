from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd,isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

N=100_000
p=[True]*(N+1)
p[0]=p[1]=False
for i in range(2,isqrt(N)+1):
    if p[i]:
        for j in range(i*i,N+1,i):
            p[j]=False
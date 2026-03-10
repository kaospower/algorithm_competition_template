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

#欧拉筛
N=100_000
p=[True]*(N+1)
p[0]=p[1]=False
primes=[]
for i in range(2,N+1):
    if p[i]:
        primes.append(i)
    for x in primes:
        if x*i>=N+1:
            break
        p[x*i]=False
        if i%x==0:
            break


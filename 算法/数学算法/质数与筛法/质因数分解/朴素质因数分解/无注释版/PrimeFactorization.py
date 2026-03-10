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

#质因数分解
def f(n):
    i,x=2,n
    p=[]
    while i<=isqrt(x):
        if x%i==0:
            p.append(i)
            while x%i==0:
                x//=i
        i+=1
    if x>1:
        p.append(x)
    return p

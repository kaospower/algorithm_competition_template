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
#时间复杂度O(根号n)
def f(n):
    #i从2开始,因为最小的素数是2
    i,x=2,n
    p=[]
    while i<=isqrt(x):
        if x%i==0:
            p.append(i)
            while x%i==0:
                x//=i
        i+=1
    #记得加上最后的素数
    if x>1:
        p.append(x)
    return p

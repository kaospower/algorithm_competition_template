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

#Miller-Rabin测试
prime=[2,3,5,7,11,13,17,19,23,29,31,37]
def witness(a,n):
    u,t=n-1,0
    while u&1==0:
        t+=1
        u>>=1
    x1=pow(a,u,n)
    for i in range(1,t+1):
        x2=pow(x1,2,n)
        if x2==1 and x1!=1 and x1!=n-1:
            return True
        x1=x2
    if x1!=1:
        return True
    return False
def p(n):
    if n<=2:
        return n==2
    if n&1==0:
        return False
    i=0
    while i<len(prime) and prime[i]<n:
        if witness(prime[i],n):
            return False
        i+=1
    return True


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

#质因数分解,O(logn)
#对于数n,分解质因数的最坏情况是2的幂,此时除法次数最多,为logn
#因此欧拉筛预处理lpf后,利用lpf分解质因数的时间复杂度为O(logn)
N=1_000_000
lpf=[0]*(N+1)
primes=[]
for i in range(2,N+1):
    if lpf[i]==0:
        lpf[i]=i
        primes.append(i)
    for x in primes:
        if (j:=x*i)>=N+1:
            break
        lpf[j]=x
        if i%x==0:
            break
def f(n):
    x=n
    p=set()
    #如果x是质数,lpf[x]=x,x除以lpf[x]之后会变成1,由于lpf[1]=0,因此会退出循环
    while lpf[x]:
        p.add(lpf[x])
        x//=lpf[x]
    return p
print(f(42))


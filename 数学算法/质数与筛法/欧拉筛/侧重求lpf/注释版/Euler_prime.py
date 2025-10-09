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
#这种写法的欧拉筛用于既需要筛质数,又需要求lpf(最小质因子)的场合
#这种写法用lpf数组代替布尔数组
#lpf[i]=x代表i的lpf是x
N=1_000_000
lpf=[0]*(N+1) #最小质因子数组
primes=[] #素数数组
for i in range(2,N+1):
    # 如果最小质因子是0,说明当前数没被筛过,是质数
    if lpf[i]==0:
        lpf[i] = i #质数的最小质因子就是它本身
        primes.append(i) #将这个数加入质数数组
    for x in primes: #p都是<=i的
        # 注意这里要写成>=N+1,如果写成>=N break,N的lpf就没被更新
        if (j:=x*i)>=N+1:
            break
        lpf[j]=x #j的lpf是x
        if i%x==0: #x是i的lpf(最小质因子)
            break

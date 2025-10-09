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
#这种写法的欧拉筛用于单纯筛质数,不需要求lpf的场合
#p[i]为True,说明i是素数,否则不是素数
#时间复杂度O(n)
#注意欧拉筛的过程中必须要求出质数数组
#和埃氏筛相比,欧拉筛时间复杂度更优
#每个合数只被筛一次,因此时间复杂度是O(n),但是常数稍大
#欧拉筛的特点是在筛质数同时会顺便处理出lpf(最小质因子)
#因此如果题目同时要判断质数以及用到lpf,使用欧拉筛会更快
N=100_000
p=[True]*(N+1)
p[0]=p[1]=False
primes=[]
for i in range(2,N+1):
    if p[i]:
        primes.append(i)
    for x in primes: #p都是<=i的
        # 注意这里要写成>=N+1,如果写成>=N break,N的lpf就没被更新
        if x*i>=N+1:
            break
        p[x*i]=False
        if i%x==0: #x是i的lpf(最小质因子)
            break


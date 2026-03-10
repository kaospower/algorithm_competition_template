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

#最小质因子lpf
#时间复杂度和埃氏筛一样O(loglogn)
N=1_000_000
lpf=[0]*(N+1) #p[i]表示i的最小质因子
#原理和埃氏筛类似
#从2开始,2的倍数2,4,6,8...等的最小质因子都是2,将它们进行标记
#然后继续查看未标记的数,重复这个操作...
#可以发现,质数的最小质因子就是它本身
#同时注意p数组下标<2时无意义
for i in range(2,N+1):
    if lpf[i]==0:
        for j in range(i,N+1,i):
            if lpf[j]==0:
                lpf[j]=i

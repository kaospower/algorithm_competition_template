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

#Floyd算法用于求全源最短路,时间复杂度O(n^3),可以求任意两点间最短距离,可以求解负权图
#start,end,weight分别代表边的起点,终点及边权,n代表点的数量,返回值为nxn数组,d[i][j]表示i~j之间最短路
def Floyd(start,end,weight,n):
    d = [[inf] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0

    for x, y, z in zip(start,end,weight):
        d[x][y] = fmin(z, d[x][y])

    for k in range(n):
        for i in range(n):
            if d[i][k] == inf:
                continue
            for j in range(n):
                d[i][j] = fmin(d[i][j], d[i][k] + d[k][j])
    return d
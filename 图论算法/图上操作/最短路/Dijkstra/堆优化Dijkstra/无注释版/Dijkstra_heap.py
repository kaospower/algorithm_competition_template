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

#单源正权最短路
def Dijkstra(edges,n,start):
    g=[[] for _ in range(n)]
    # 默认是有向图,如果是无向图需要额外连反向边
    for x,y,w in edges:
        g[x].append((y,w))
    dis=[inf]*n
    dis[start]=0
    pq=[(0,start)]
    while pq:
        dis_u,u=heappop(pq)
        if dis_u>dis[u]:
            continue
        for v,w in g[u]:
            dis_v=dis_u+w
            if dis_v<dis[v]:
                dis[v]=dis_v
                heappush(pq,(dis_v,v))
    return dis


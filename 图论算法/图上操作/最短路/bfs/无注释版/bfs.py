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

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

#无权最短路
def bfs(start,edges,n):
    g=[[] for _ in range(n)]
    for x,y in edges:
        g[x].append(y)
        g[y].append(x)
    dis=[inf]*n
    dis[start]=0
    q=deque([start])
    level=1
    while q:
        size=len(q)
        for _ in range(size):
            u=q.popleft()
            for v in g[u]:
                if level<dis[v]:
                    q.append(v)
                    dis[v]=level
        level+=1
    return dis


from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd
from collections import defaultdict, deque

# from sortedcontainers import SortedList
# from itertools import pairwise

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

def topo(edges, n):
    indegree = [0] * n
    g = [[] for _ in range(n)]
    for x, y in edges:
        g[x].append(y)
        indegree[y] += 1
    arr = [i for i, x in enumerate(indegree) if x == 0]
    q = deque(arr)
    res = []
    while q:
        cur = q.popleft()
        res.append(cur)
        for x in g[cur]:
            indegree[x] -= 1
            if indegree[x] == 0:
                q.append(x)
    if len(res) != n:
        return -1

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

#单源正权最短路

#权值更新函数,根据题意编写
def g(dis_u,w):
    pass

def Dijkstra(grid, sx, sy):
    move = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    n, m = len(grid), len(grid[0])
    dis = [[inf] * m for _ in range(n)]
    dis[sx][sy] = 0
    pq = [(0, sx, sy)]
    while pq:
        dis_u, x, y = heappop(pq)
        if dis_u > dis[x][y]:
            continue
        for dx, dy in move:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m:
                dis_v = g(dis_u,grid[nx][ny]) #根据题意修改
                if dis_v < dis[nx][ny]:
                    dis[nx][ny] = dis_v
                    heappush(pq, (dis_v, nx, ny))
    return dis

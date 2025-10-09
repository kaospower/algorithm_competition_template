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
#注意Dijkstra优先松弛权重最短的边,因此当一个点首次被弹出时,其对应的距离一定就是源点到它的最短路
#edges是边集(包含权重),默认下标从0开始,如果下标从1开始,n初始化成1,start表示源点
def Dijkstra(edges,n,start):
    g=[[] for _ in range(n)]
    # 默认是有向图,如果是无向图需要额外连反向边
    for x,y,w in edges:
        g[x].append((y,w))
    dis=[inf]*n #存储源点到每个点的最短路
    dis[start]=0
    pq=[(0,start)] #将源点入堆,第一维记录源点到当前点的距离,第二维记录当前点
    while pq:
        dis_u,u=heappop(pq)
        #如果dis_u大于源点到u的历史最短距离,说明u入过堆,continue,防止重复入堆
        if dis_u>dis[u]:
            continue
        for v,w in g[u]:
            dis_v=dis_u+w
            if dis_v<dis[v]:
                dis[v]=dis_v
                heappush(pq,(dis_v,v))
    return dis


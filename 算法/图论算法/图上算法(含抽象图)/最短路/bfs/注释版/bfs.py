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

#start代表起点,edges代表边集,n代表点数
def bfs(start,edges,n):
    g=[[] for _ in range(n)]
    for x,y in edges:
        g[x].append(y)
        g[y].append(x)
    #dis数组记录起点到其他点的最短路,这里默认起点是0
    #这里的dis数组同时承担visit数组的功能,如果一个点之前访问过,那么后续再试图访问它时,一定会用更大的距离
    #由于使用了类似Dijkstra的最优性剪枝,只访问最短路,因此不会重复访问节点
    dis=[inf]*n
    dis[start]=0
    q=deque([start])
    #level记录当前遍历到第几层,即距离
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

#模版题(https://leetcode.cn/problems/the-time-when-the-network-becomes-idle/)
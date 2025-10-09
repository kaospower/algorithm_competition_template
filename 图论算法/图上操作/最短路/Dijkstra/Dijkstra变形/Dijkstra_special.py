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

#特殊情况下Dijkstra可以维护单源最长路
#权重累积为乘法且权值在0~1之间的最长路问题可以用类似Dijkstra的方法解决,不过要用大根堆

#回顾原始Dijkstra求最短路,它利用了加法不会变小的性质,优先松弛短的边,从而得到最短路
#而本题边权全是介于0~1之间的数,且为乘法,一个数乘上<=1的数不可能变大,只会不变或变小
#因此本题的乘法不会变大,贪心地优先松弛大的边,就可以得到最长路
#由于是维护最大边,维护一个大根堆,使用Dijkstra即可
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start_node: int, end_node: int) -> float:
        def Dijkstra(start):
            g=[[] for _ in range(n)]
            for (x,y),w in zip(edges,succProb):
                g[x].append((y,w))
                g[y].append((x,w))
            dis=[0]*n
            dis[start]=1
            pq=[(-1,start)]
            #python大根堆需要将数值取反
            while pq:
                dis_u,u=heappop(pq)
                if u==end_node:
                    return -dis_u
                if -dis_u<dis[u]:
                    continue
                for v,w in g[u]:
                    dis_v=-dis_u*w
                    if dis_v>dis[v]:
                        dis[v]=dis_v
                        heappush(pq,(-dis_v,v))
            return 0
        return Dijkstra(start_node)

#模版题:1514(https://leetcode.cn/problems/path-with-maximum-probability/)

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

#本题特殊在bfs遍历的不是点,而是集合
#处理方式其实和普通bfs一样,只不过要多一重循环遍历集合中的所有点
#标记数组也需要两个,一个标记访问过的集合,一个标记访问过的点
#这里对于访问过的集合,直接置成None,dis哈希表则承担鉴别访问过的点的功能

#815(https://leetcode.cn/problems/bus-routes/)
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        # 将站映射到公交,然后再将公交映射到站
        # bfs的每层将同一个路线中没访问过的站全部加入队列
        g = defaultdict(list)
        for i, route in enumerate(routes):
            for x in route:
                g[x].append(i)
        # 起点或者终点不在公交路线中,如果起点和终点不等,返回-1,否则返回0
        if source not in g or target not in g:
            return -1 if source != target else 0

        dis = {source: 0}
        q = deque([source])
        while q:
            u = q.popleft()
            for i in g[u]:
                if routes[i]:
                    for v in routes[i]:
                        if v not in dis:
                            dis[v] = dis[u] + 1
                            q.append(v)
                    routes[i] = None
        return dis.get(target, -1)

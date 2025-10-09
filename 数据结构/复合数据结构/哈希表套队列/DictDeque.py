from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

#哈希表套队列+二分
class Router:

    def __init__(self, memoryLimit: int):
        self.d=defaultdict(deque)
        self.q=deque()
        self.s=set()
        self.size=0
        self.maxsize=memoryLimit
    def addPacket(self, source: int, destination: int, timestamp: int) -> bool:
        if (source,destination,timestamp) in self.s:
            return False
        if self.size==self.maxsize:
            self.forwardPacket()
        self.d[destination].append((timestamp,source))
        self.s.add((source,destination,timestamp))
        self.q.append([source,destination,timestamp])
        self.size+=1
        return True

    def forwardPacket(self) -> List[int]:
        if self.q:
            source,destination,timestamp=self.q.popleft()
            self.d[destination].popleft()
            self.s.remove((source,destination,timestamp))
            self.size-=1
            return [source,destination,timestamp]
        return []

    def getCount(self, destination: int, startTime: int, endTime: int) -> int:
        a=self.d[destination]
        l=bisect_left(a,(startTime,))
        r=bisect_left(a,(endTime+1,))-1
        return r-l+1

#示例题:3508(https://leetcode.cn/problems/implement-router/)
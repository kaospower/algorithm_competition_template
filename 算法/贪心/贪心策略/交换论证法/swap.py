from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache, reduce
from math import gcd, isqrt
from operator import and_, or_, xor, add, mul
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

# leetcode 1665(https://leetcode.cn/problems/minimum-initial-energy-to-finish-tasks/description/)
#交换论证法
class Solution:
    def minimumEffort(self, tasks: List[List[int]]) -> int:
        tasks.sort(key=lambda p:-(p[1]-p[0]))
        ans=s=0
        for a,b in tasks:
            #初始能量为ans,s为累计消耗能量,b为任务最低能量,ans-s>=b,即ans>=s+b
            ans=max(ans,s+b)
            s+=a
        return ans
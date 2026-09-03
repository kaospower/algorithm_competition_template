from typing import List, Optional
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


# 模版题 3796(https://leetcode.cn/problems/find-maximum-value-in-a-constrained-sequence/)
# 两次扫描,第一次正向遍历确定左侧约束,第二次反向遍历确定反向约束
class Solution:
    def findMaxVal(self, n: int, restrictions: List[List[int]], diff: List[int]) -> int:
        mx=[inf]*n
        a=[0]*n
        for i,v in restrictions:
            mx[i]=v
        for i,v in enumerate(diff):
            a[i+1]=min(a[i]+v,mx[i+1])
        for i in range(n-2,0,-1):
            a[i]=min(a[i],a[i+1]+diff[i])
        return max(a)
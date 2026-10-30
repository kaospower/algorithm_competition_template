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

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y

#分治法核心是将原问题分解为两个子问题,递归求解
#实际实现方式类似快速排序

#1545(https://leetcode.cn/problems/find-kth-bit-in-nth-binary-string)  
class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        def f(n,k):
            if n == 1:
                return '0'
            if k == 1 << (n - 1):
                return '1'
            if k < 1 << (n - 1):
                return f(n - 1, k)
            res = f(n - 1, (1 << n) - k)
            return '0' if res == '1' else '1'
        return f(n,k)

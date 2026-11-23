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


#本质仍是前缀和变式，但是需要特判0

#前缀积
# leetcode 1352(https://leetcode.cn/problems/product-of-the-last-k-numbers/)
class ProductOfNumbers:

    def __init__(self):
        self.arr=[1]

    def add(self, num: int) -> None:
        if num:
            self.arr.append(self.arr[-1]*num)
        else:
            self.arr=[1]

    def getProduct(self, k: int) -> int:
        if len(self.arr)<=k:
            return 0
        return self.arr[-1]//self.arr[-k-1]
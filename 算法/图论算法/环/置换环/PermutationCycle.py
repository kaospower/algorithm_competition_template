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

#置换环是用来求解数组排序元素间所需最小交换次数这类问题。

# leetcode 3551(https://leetcode.cn/problems/minimum-swaps-to-sort-by-digit-sum/description/)
#置换环+并查集
#对于数字互不相同的数组的两个不同排列,将一个排列变成另一个的最少交换次数即为
#数组长度减去将对应下标的数字形成的映射连成边的图中的环的数目
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        n=len(nums)
        arr=sorted(range(n),key=lambda i:(sum(map(int,str(nums[i]))),nums[i]))
        u=UnionFind(n)
        for i,j in enumerate(arr):
            u.union(i,j)
        return n-u.cc

class UnionFind:
    def __init__(self, n):
        self.father = list(range(n))
        self.cc = n

    def find(self, i):
        if i != self.father[i]:
            self.father[i] = self.find(self.father[i])
        return self.father[i]

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx == fy:
            return False
        self.father[fx] = fy
        self.cc -= 1
        return True

    def reset(self):
        n = len(self.father)
        self.father = list(range(n))
        self.cc = n

    def separate(self, x, y):
        self.father[x], self.father[y] = x, y
        self.cc += 1

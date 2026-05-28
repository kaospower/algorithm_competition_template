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


# RMQ预处理ST时间为O(nlogn),单次查询时间复杂度O(1),查询速度比线段树快,但是由于初始化时间复杂度较大,因此只有海量查询时才优于线段树
# 第一维时间复杂度O(n),第二维logn,因此预处理总时间复杂度O(nlogn)
# f[i][j]表示从i开始长度是2^j的区间最值
# 设待查询区间是[l,r],长度是len,寻找最大的k,使得2^k<=len,即log(len)下取整
# 查询时,寻找两段长为k的区间,将待查询区间[l,r]覆盖
# 注意ST表是静态算法,不支持修改

# RMQ求区间最小值,本模版同时求出了最小值对应的下标,当有多个值一样时,取最小下标
class RMQ:
    def __init__(self, nums):
        n = len(nums)
        # m为logn上取整
        m = n.bit_length()
        # st表
        f = [[0] * m for _ in range(n)]
        # 预处理
        for j in range(m):
            for i in range(n + 1 - (1 << j)):
                # j=0时,2^j=1,只有一个数
                if j == 0:
                    f[i][j] = (nums[i], i)
                    # 通常情况下赋值是f[i][j]=nums[i]
                else:
                    f[i][j] = self.merge(f[i][j - 1], f[i + (1 << j - 1)][j - 1])
        self.n = n
        self.m = m
        self.f = f

    # 根据实际需要修改,如果是求最大就改成Max
    def merge(self, a, b):
        return Min(a, b)

    def query(self, l, r):
        size = r - l + 1
        k = size.bit_length() - 1  # log下取整
        return self.merge(self.f[l][k], self.f[r - (1 << k) + 1][k])


# 使用示例,1673(https://leetcode.cn/problems/find-the-most-competitive-subsequence/description/)
class Solution:
    def mostCompetitive(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        rmq = RMQ(nums)
        ans = [0] * k
        idx = -1
        for i in range(k):
            v, idx = rmq.query(idx + 1, n - 1 - (k - i - 1))
            ans[i] = v
        return ans

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


# leetcode 122(https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)
#状态机dp
#一般定义f[i][j]表示前缀a[:i]在状态j下的最优值,j一般很小
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        f0,f1=0,-inf #f0,f1分别表示到第i天为止不持有,持有股票的最大收益
        for x in prices:
            f0,f1=max(f0,f1+x),max(f1,f0-x)
        #由于到最后不持有股票的收益一定大于持有股票的收益,因此返回f0即可
        return f0
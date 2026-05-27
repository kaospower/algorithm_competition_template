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

# leetcode 3121(https://leetcode.cn/problems/count-the-number-of-special-characters-ii/description/)
# 位集优化
# 小写字母a ascii码值从97开始,减去96(2^6+2^5)后刚好从1开始
# 大写字母a ascii码值从65开始,减去64(2^6)后刚好从1开始
class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower=upper=invalid=0
        for x in map(ord,word):
            bit=1<<(x&31)
            #小写
            if x&32:
                lower|=bit
                if upper&bit:
                    invalid|=bit
            else:
                upper|=bit
        return (lower&upper&~invalid).bit_count()


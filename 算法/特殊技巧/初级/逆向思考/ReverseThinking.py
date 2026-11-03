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


# leetcode 780(https://leetcode.cn/problems/reaching-points/)

# 正难则反,逆向思考
# 从终点考虑
# 辗转相除直到tx,ty中的一个小于等于起点坐标
# 如果两个都比起点坐标小,不可达
# 如果tx==sx,看ty能否通过辗转相除到达,否则,考虑tx
class Solution:
    def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
        while tx > sx and ty > sy:
            if tx > ty:
                tx %= ty
            else:
                ty %= tx
        if tx < sx or ty < sy:
            return False
        return (ty - sy) % tx == 0 if tx == sx else (tx - sx) % ty == 0

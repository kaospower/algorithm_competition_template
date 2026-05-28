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
#状态机
class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        ans = 0
        state = [0] * 27
        for c in map(ord, word):
            x = c & 31  # 转成数字 1~26
            if c & 32:  # 小写字母
                if state[x] == 0:
                    state[x] = 1
                elif state[x] == 2:  # 大写的后面不能有小写
                    state[x] = -1
                    ans -= 1
            else:  # 大写字母
                if state[x] == 0:  # 还没遇到小写，就先遇到大写了
                    state[x] = -1
                elif state[x] == 1:
                    state[x] = 2
                    ans += 1
        return ans
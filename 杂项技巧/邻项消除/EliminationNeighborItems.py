from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, lcm,isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y


#邻项消除通常用栈来实现
#模版题:2187(https://leetcode.cn/problems/replace-non-coprime-numbers-in-array/description/)
def f(nums):
    st = []
    for x in nums:
        while st and gcd(st[-1], x) > 1:
            x = lcm(st.pop(), x)
        st.append(x)
    return st

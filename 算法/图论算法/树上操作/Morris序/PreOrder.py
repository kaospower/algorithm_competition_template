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


def f(root):
    ans = []
    cur = root
    mostRight = None
    while cur is not None:
        mostRight = cur.left
        if mostRight is not None:
            while mostRight.right is not None and mostRight.right != cur:
                mostRight = mostRight.right
            if mostRight.right is None:  # 第一次到达
                # 有左树的节点第一次到达时收集
                ans.append(cur.val)
                mostRight.right, cur = cur, cur.left
                continue
            else:  # 第二次到达
                mostRight.right = None
        else:
            # 没有左树的节点直接收集
            ans.append(cur.val)
        cur = cur.right
    return ans

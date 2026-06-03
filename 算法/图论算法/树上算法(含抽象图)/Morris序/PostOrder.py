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

    def reverse(root):
        pre = nxt = None
        while root is not None:
            nxt, root.right = root.right, pre
            pre, root = root, nxt
        return pre

    def collect(root):
        tail = reverse(root)
        cur = tail
        while cur is not None:
            ans.append(cur.val)
            cur = cur.right
        reverse(tail)

    cur = root
    mostRight = None
    while cur is not None:
        mostRight = cur.left
        if mostRight is not None:
            while mostRight.right is not None and mostRight.right != cur:
                mostRight = mostRight.right
            if mostRight.right is None:  # 第一次到达
                mostRight.right, cur = cur, cur.left
                continue
            else:  # 第二次到达
                mostRight.right = None
                collect(cur.left)
        cur = cur.right
    collect(root)
    return ans

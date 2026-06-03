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


# Morris遍历解决的问题
# 二叉树三种序的遍历,时间复杂度为O(n),额外空间复杂度为O(1)
# 没有左子树的节点只到达一次,有左子树的节点会到达两次
# Morris遍历过程
# 1.开始时cur来到头节点,cur为空时过程停止
# 2.如果cur没有左孩子,cur向右移动
# 3.如果cur有左孩子,找到cur左子树的最右节点mostRight
# A.如果mostRight的右指针指向空,让其指向cur,然后cur向左移动
# B.如果mostRight的右指针指向cur,让其指向null,然后cur向右移动
def Morris(root):
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
        cur = cur.right

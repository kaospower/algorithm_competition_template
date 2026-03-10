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

#非递归实现二叉树先序遍历
#压栈时先压右右节点,后压左节点
def f(root):
    ans = []
    if root:
        st = [root]
        while st:
            root = st.pop()
            ans.append(root.val)
            if root.right:
                st.append(root.right)
            if root.left:
                st.append(root.left)
    return ans


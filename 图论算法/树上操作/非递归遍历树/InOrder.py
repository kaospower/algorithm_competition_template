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

#非递归实现二叉树中序遍历
#1.子树左边界进栈
#2.栈弹出节点打印,节点右树,重复步骤1
#直到没子树且栈空时停止
def f(root):
    ans=[]
    if root:
        st=[]
        while st or root:
            if root:
                st.append(root)
                root=root.left
            else:
                root=st.pop()
                ans.append(root.val)
                root=root.right
    return ans
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

#参考先序遍历,压栈时先压左,后压右,可以得到反先序(即中右左)
#将这个顺序反转就可得到后序遍历(左右中)
#两个栈实现
def f(root):
    ans = []
    if root:
        st = [root]
        collect=[]
        while st:
            root = st.pop()
            collect.append(root)
            if root.left:
                st.append(root.left)
            if root.right:
                st.append(root.right)
        while collect:
            ans.append(collect.pop().val)
    return ans

#一个栈实现
def f(root):
    ans = []
    if root:
        st = [root]
        #如果始终没有打印过节点,root就是头节点
        #一旦打印过节点,root就变成打印节点
        #之后root的含义:上一次打印的节点
        while st:
            cur = st[-1]
            if cur.left and root!=cur.left and root!=cur.right:
                st.append(cur.left)
            elif cur.right and root!=cur.right:
                st.append(cur.right)
            else:
                ans.append(cur.val)
                root=st.pop()
    return ans
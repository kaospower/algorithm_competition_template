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

#动态点分治
#动态点分治,就是在查询的基础上,又有修改操作

#点分树是保证修改/查询都能快速完成的关键

#点分树
#1.点分治的过程中,每次求出一颗子树的重心
#2.记录每个重心的上级重心(centfather数组),等同于记录了一棵重构的树
#3.这棵重构的树叫点分树,高度为O(logn)
#4.当然可以完整记录点分树,不过一般情况下,只记录上级重心就够了

#点分树
#1.点分树上的一个子树,一定是原树的一个联通块
#2.点分树上两点的LCA,一定在原树两点的简单路径上
#3.除此之外,点分树和原树的联系很少
class Solution:
    pass


s = Solution()

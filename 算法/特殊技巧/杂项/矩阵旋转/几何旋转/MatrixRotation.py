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

# leetcode 1861(https://leetcode.cn/problems/rotating-the-box/)

# 先用双指针确定位置,然后旋转
class Solution:
    def rotateTheBox(self, boxGrid: List[List[str]]) -> List[List[str]]:
        m=len(boxGrid[0])
        for i,row in enumerate(boxGrid):
            #k记录石头位置
            k=m-1
            for j in range(m-1,-1,-1):
                if row[j]=='*':
                    k=j-1
                elif row[j]=='#':
                    row[j],row[k]=row[k],row[j]
                    k-=1
        #注意和转置的区别,转置不需要先反转矩阵
        return list(zip(*reversed(boxGrid)))

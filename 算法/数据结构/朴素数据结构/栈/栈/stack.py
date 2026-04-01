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

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

#leetcode 2751(https://leetcode.cn/problems/robot-collisions/description/)
# 栈
class Solution:
    def survivedRobotsHealths(self, positions: List[int], healths: List[int], directions: str) -> List[int]:
        n = len(positions)
        ids = sorted(range(n), key=lambda i: positions[i])
        st = []
        for i in ids:
            d, h = directions[i], healths[i]
            # 当前向右
            if d == 'R':
                st.append([i, h, d])
            else:
                # 当前向左,栈顶向右且健康值小于当前
                while st and st[-1][2] == 'R' and st[-1][1] < h:
                    st.pop()
                    h -= 1
                # 当前向左,栈顶向右且健康值等于当前
                if st and st[-1][2] == 'R' and st[-1][1] == h:
                    st.pop()
                # 当前向左,栈顶向右且健康值大于当前
                elif st and st[-1][2] == 'R' and st[-1][1] > h:
                    st[-1][1] -= 1
                # 其他情况,不会发生碰撞,直接入栈
                else:
                    st.append([i, h, d])

        st.sort()
        return [h for i, h, d in st]

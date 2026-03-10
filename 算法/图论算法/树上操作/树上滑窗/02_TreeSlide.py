from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

#示例题:3486(https://leetcode.cn/problems/longest-special-path-ii/description/)
#树上滑窗,本题是3425题进阶
class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n=len(edges)+1
        g=[[] for _ in range(n)]
        for x,y,z in edges:
            g[x].append((y,z))
            g[y].append((x,z))
        ans=(-inf,0)
        pos={} #注意,下文中所有涉及到深度(下标)的位置都是实际深度+1
        dis=[0]
        #本题和3425区别是,允许路径中一个颜色出现两次
        #last1表示窗口内出现两次元素的靠左位置
        #last2表示窗口内当前颜色上一次出现的位置
        #每轮更新答案之前,更新窗口上边界
        def f(u,fa,top,last1):
            color=nums[u]
            last2=pos.get(color,0)
            #top更新分两种情况
            #一种是abba,即当前元素和上一次出现的位置之间存在出现两次的元素
            #这时top更新为last2
            #另一种是aabb,即当前元素和上一次出现的位置之间没有元素出现两次
            #这是top更新为last1
            #综上,last1,last2取最小值作为左边界
            top=max(top,min(last1,last2))
            nonlocal ans
            ans=max(ans,(dis[-1]-dis[top],top-len(dis)))
            pos[color]=len(dis)
            for v,w in g[u]:
                if v==fa:continue
                dis.append(dis[-1]+w)
                #last1维护的是窗口内出现两次的元素的位置,因此必须时刻保持最新
                #因此新的last1必须是更大的
                #在更新时,同样有两种情况,一种是当前元素上一次出现的位置last2很靠右,因此用last2更新
                #还有一种是上一次出现是很久之前,last1保持不变
                #综上,last1,last2取最大即可
                f(v,u,top,max(last1,last2))
                dis.pop()
            pos[color]=last2
        f(0,-1,0,0)
        return [ans[0],-ans[1]]
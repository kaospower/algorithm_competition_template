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

#示例题:3425:https://leetcode.cn/problems/longest-special-path/
#树上滑窗
class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        #建图
        n=len(edges)+1
        g=[[] for _ in range(n)]
        for x,y,z in edges:
            g[x].append((y,z))
            g[y].append((x,z))
        #ans第一维记录最长路径,第二维记录最少点数
        #最少点是实际值取反,为了可以通过max一次求最大路径和最小点数
        ans=(-inf,0)
        #记录某种颜色上次出现的深度,注意是实际深度+1,或者可以说是滑窗上边界的深度
        pos={}
        #dis本质是一个栈,记录了从根到当前点的距离,类似前缀和,用数组记录是因为通过len(dis)可以顺便求出当前深度
        dis=[0]
        #top参数记录滑窗上边界
        def f(u,fa,top):
            color=nums[u] #当前点颜色
            last=pos.get(color,0) #color上一次出现的深度
            #更新滑窗上边界,取最大值是因为会出现01210这种情况
            #0上一次出现的位置比1上一次出现的位置小,但是显然要取深度更大的
            top=max(top,last)
            nonlocal ans
            #dis[-1]-dis[top]可以算出路径长度,即根到下边界距离-根到上边界距离
            #len(dis)-top即当前点深度-当前点上一次出现的深度,即滑窗内点数,取反是可以通过max更新最小值
            ans=max(ans,(dis[-1]-dis[top],top-len(dis)))
            #更新当前点深度
            pos[color]=len(dis)
            for v,w in g[u]:
                if v==fa:continue
                #将根到孩子节点距离加入数组
                dis.append(dis[-1]+w)
                f(v,u,top)
                dis.pop()
            pos[color]=last #回溯
        f(0,-1,0)
        return [ans[0],-ans[1]]

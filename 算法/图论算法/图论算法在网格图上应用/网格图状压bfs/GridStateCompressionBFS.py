from typing import List, Optional
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


# 网格图状压bfs
# 这类题目由于不满足无后效性,因此不能用dp,只能考虑bfs/Dijkstra等
# 本题经过分析可知需要多次经过同一个格子,如何避免无限循环呢,这就需要记录经过格子时的状态
# 本题中即为能量值和收集的垃圾数,只有这两个值不同时,反复经过相同的格子才是有意义的
# 由于垃圾数较多,因此考虑用状态压缩记录收集的垃圾状态
# 因此需要用四维数组记录横纵坐标,剩余能量值以及收集垃圾的状态
# 本题的难点在于和普通bfs相比,vis数组多了两个记录其他信息的维度,这是很难想到的
# 本题的一个优化点是,当垃圾状态和位置相同时,只记录能量最大的状态,因为更小的能量是不可能有更短的步数的,这是基于贪心的优化思想
# 模版题leetcode 3568(https://leetcode.cn/problems/minimum-moves-to-clean-the-classroom/)
class Solution:
    def minMoves(self, classroom: List[str], energy: int) -> int:
        n,m=len(classroom),len(classroom[0])
        garbage=[[0]*m for _ in range(n)]
        cnt=sx=sy=0
        for i,row in enumerate(classroom):
            for j,x in enumerate(row):
                if x=='L':
                    garbage[i][j]=1<<cnt
                    cnt+=1
                elif x=='S':
                    sx,sy=i,j
        if cnt==0:
            return 0

        dirs=(0,1),(0,-1),(1,0),(-1,0)
        vis=[[[[False]*(1<<cnt) for _ in range(energy+1)] for _ in range(m)] for _ in range(n)]
        vis[sx][sy][energy][0]=True
        q=deque([(sx,sy,energy,0)])
        s=(1<<cnt)-1

        dis=0
        while q:
            size=len(q)
            for _ in range(size):
                x,y,e,g=q.popleft()
                if g==s:
                    return dis
                if e==0:
                    continue
                for dx,dy in dirs:
                    nx,ny=x+dx,y+dy
                    if 0<=nx<n and 0<=ny<m and classroom[nx][ny]!='X':
                        ne=energy if classroom[nx][ny]=='R' else e-1
                        ng=g|garbage[nx][ny]
                        if not vis[nx][ny][ne][ng]:
                            vis[nx][ny][ne][ng]=True
                            q.append([nx,ny,ne,ng])
            dis+=1
        return -1

# 优化,去掉vis数组中的能量维度,本质是一种基于贪心的剪枝
# 当位置和收集的垃圾数量相同时,只保留能量更大的状态
# 这样大大减少了需要重复遍历的状态数
class Solution:
    def minMoves(self, classroom: List[str], energy: int) -> int:
        n,m=len(classroom),len(classroom[0])
        garbage=[[0]*m for _ in range(n)]
        cnt=sx=sy=0
        for i,row in enumerate(classroom):
            for j,x in enumerate(row):
                if x=='L':
                    garbage[i][j]=1<<cnt
                    cnt+=1
                elif x=='S':
                    sx,sy=i,j
        if cnt==0:
            return 0

        dirs=(0,1),(0,-1),(1,0),(-1,0)
        vis=[[[-1]*(1<<cnt) for _ in range(m)] for _ in range(n)]
        vis[sx][sy][0]=energy
        q=deque([(sx,sy,energy,0)])
        s=(1<<cnt)-1

        dis=0
        while q:
            size=len(q)
            for _ in range(size):
                x,y,e,g=q.popleft()
                if g==s:
                    return dis
                if e==0:
                    continue
                for dx,dy in dirs:
                    nx,ny=x+dx,y+dy
                    if 0<=nx<n and 0<=ny<m and classroom[nx][ny]!='X':
                        ne=energy if classroom[nx][ny]=='R' else e-1
                        ng=g|garbage[nx][ny]
                        if ne>vis[nx][ny][ng]:
                            vis[nx][ny][ng]=ne
                            q.append([nx,ny,ne,ng])
            dis+=1
        return -1
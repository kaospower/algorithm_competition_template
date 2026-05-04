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

#矩阵原地旋转问题

#解法1,顺时针旋转90度等价于转置一次+行翻转一次
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n=len(matrix)
        for i,row in enumerate(matrix):
            for j in range(i+1,n):
                matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
            row.reverse()

#解法2,将矩阵均匀分成四块后(奇数阶方阵需要扣除中心点),原问题转化成四点轮换问题
#根据观察可以发现,顺时针旋转90度会将(i,j)变为(j,n-1-i),逆时针旋转90度会将(i,j)变为(n-1-j,i)
#题目要求顺时针旋转90度,那么旋转之后的矩阵的某个点的值即为将该点逆时针旋转90度后,得到的点在原矩阵中对应的值
#按顺序顺时针枚举四个点,然后将它们同时置换为逆时针旋转90度对应的点即可
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                matrix[i][j],matrix[j][n-1-i],matrix[n-1-i][n-1-j],matrix[n-1-j][i]=\
                matrix[n-1-j][i],matrix[i][j],matrix[j][n-1-i],matrix[n-1-i][n-1-j]

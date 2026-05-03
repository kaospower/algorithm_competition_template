from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise,chain

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

"""
kmp模版2
s代表文本串,p代表模式串
返回模式串p在文本串中的所有匹配位置,如果不存在匹配位置返回空数组
时间复杂度O(n+m)
"""
def kmp(s, p):
    # n代表s长度,m代表p长度
    n, m = len(s), len(p)
    ne = [0] * m  # ne[i]表示模式串p在0...i范围前缀和后缀的最大匹配长度

    # 求解next数组的过程,注意next数组是对模式串p求的
    j = 0
    for i in range(1, m):
        while j and p[j] != p[i]:
            # 如果不等,j跳转到ne[j-1]
            j = ne[j - 1]
        if p[j] == p[i]:
            j += 1
        ne[i] = j

    # KMP匹配过程
    # p[0...j]和s[0...i]匹配,如果发现p[j]和s[i]不匹配,就递归地将p[ne[j-1]]和s[i]匹配
    # 由于ne[i]存储的是长度,因此p在下标0~ne[j-1]-1这ne[j-1]长度是一定匹配的,此时检验p[ne[j-1]]是否和s[i]匹配即可
    # pos存放所有匹配位置
    pos=[]
    j = 0
    for i, v in enumerate(s):
        while j and p[j] != v:
            j = ne[j - 1]
        if p[j] == v:
            j += 1
        # 匹配成功,将s中包含p的开头位置加入pos
        if j == m:
            pos.append(i-m+1)
            j=ne[j-1]
    return pos
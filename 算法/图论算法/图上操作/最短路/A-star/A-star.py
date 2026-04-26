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

#指定源点,目标点,求源点到目标点的最短距离
#增加了当前点到终点的预估函数
#在堆中根据从源点出发到达当前点的距离+当前点到终点的预估距离来进行排序
#预估函数要求:当前点到终点的预估距离<=当前点到终点的真实最短距离
#预估终点距离经常选择:曼哈顿距离,欧式距离,对角线距离


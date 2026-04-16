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

#例1
#对数组a,接下来有m条操作
#将所有模x==y的下标对应的值累加起来打印出来

#解法
#x<=sqrt(n),利用预处理信息
#x>sqrt(n),暴力枚举




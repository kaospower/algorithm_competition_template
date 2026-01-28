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


def monotonic_stack(arr):
    n=len(arr)
    right=[-1]*n
    st=[]
    for i,t in enumerate(arr):
        while st and t>arr[st[-1]]:
            right[st.pop()]=i
        st.append(i)
    return right

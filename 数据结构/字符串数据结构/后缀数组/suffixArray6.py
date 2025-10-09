#更快的python基于诱导排序实现的后缀数组板子
#见1163题题解(https://leetcode.cn/problems/last-substring-in-lexicographical-order/solutions/3028019/python-hou-zhui-shu-zu-you-dao-pai-xu-by-4cbo/)
from typing import List
def inducedSort(s: List[int], sa1: List[int], t: List[int], m=26):
    n = len(s)
    cnt = [0] * m
    for i in range(n):
        cnt[s[i]] += 1
    for i in range(1, m):
        cnt[i] += cnt[i - 1]
    start, end = [0] + cnt[:-1], cnt[:]
    sa = [-1] * n + [n]
    for x in reversed(sa1[1:]):
        cnt[s[x]] -= 1
        sa[cnt[s[x]]] = x
    for i in range(-1, n):
        if sa[i] > 0:
            c = sa[i] - 1
            if not t[c]:
                sa[start[s[c]]] = c
                start[s[c]] += 1
    for i in range(n - 1, -1, -1):
        if sa[i] > 0:
            c = sa[i] - 1
            if t[c]:
                end[s[c]] -= 1
                sa[end[s[c]]] = c
    return sa[:-1]

def SA_IS(s: List[int], m=26):
    n = len(s)
    t = [False] * (n + 1)
    for i in reversed(range(n - 1)):
        t[i] = t[i + 1] if s[i] == s[i + 1] else (s[i] < s[i + 1])
    critical = list()
    for i in range(1, n):
        if t[i] and not t[i - 1]:
            critical.append(i)
    nc = len(critical)
    index = [-1] * n + [n]
    for i, x in enumerate(critical):
        index[x] = i
    sa0 = inducedSort(s, [n] + critical, t, m)
    s1 = [0] * (nc + 1)
    critical.append(n)
    last, p, repeat = "", 0, False
    for x in sa0:
        if index[x] >= 0:
            c = s[x : critical[index[x] + 1]]
            if c != last:
                p += 1
                last = c
            else:
                repeat = True
            s1[index[x]] = p
    if repeat:
        sa1 = [critical[x] for x in SA_IS(s1, p + 1)]
    else:
        sa1 = [n] + [x for x in sa0 if index[x] >= 0]
    return inducedSort(s, sa1, t, m)

def suffixArray(s: str) -> (List[int], List[int], List[int]):
    n, k = len(s), 0
    sa = SA_IS([ord(x) - 97 for x in s])
    rk = [0] * n
    for i in range(n):
        rk[sa[i]] = i
    height = [0] * n
    s += '#'
    for i in range(n):
        if rk[i]:
            if k > 0:
                k -= 1
            while s[i + k] == s[sa[rk[i] - 1] + k]:
                k += 1
            height[rk[i]] = k
    return rk, sa, height

class Solution:
    def lastSubstring(self, s: str) -> str:
        rk, sa, height = suffixArray(s)
        mh = max(rk)
        start = rk.index(mh)
        return s[start:]

"""
kmp模版
时间复杂度O(n+m)
s代表文本串,p代表模式串
"""
def kmp(s, p):
    # n代表s长度,m代表p长度
    n, m = len(s), len(p)
    ne = [0] * m  # ne[i]表示0...i范围前缀和后缀的最大匹配长度

    # TODO:求解next数组的过程,注意next数组是对模式串p求的
    j = 0
    for i in range(1, m):
        while j and p[j] != p[i]:
            # 如果不等,j跳转到ne[j-1]
            j = ne[j - 1]
        if p[j] == p[i]:
            j += 1
        ne[i] = j

    # TODO:KMP匹配过程
    # p[0...j]和s[0...i]匹配,如果发现p[j]和s[i]不匹配,就递归地将p[ne[j-1]]和s[i]匹配
    j = 0
    for i, v in enumerate(s):
        while j and p[j] != v:
            j = ne[j - 1]
        if p[j] == v:
            j += 1
        # 匹配成功,返回在s中包含p的最左开头位置
        if j == m:
            return i - m + 1
    # 无法成功匹配,返回-1
    return -1
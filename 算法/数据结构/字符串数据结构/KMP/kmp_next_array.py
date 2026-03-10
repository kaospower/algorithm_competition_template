"""
单纯求解next数组
"""
def ne(p):
    m = len(p)
    ne = [0] * m  # ne[i]表示0...i范围前缀和后缀的最大匹配长度

    # TODO:求解next数组的过程
    j = 0
    for i in range(1, m):
        while j and p[j] != p[i]:
            # 如果不等,j跳转到ne[j-1]
            j = ne[j - 1]
        if p[j] == p[i]:
            j += 1
        ne[i] = j
    return ne
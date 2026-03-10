# 按秩合并,带标签和返回值的并查集,存储形式为数组,find函数用递归实现
class UnionFind:
    def __init__(self, n):
        self.father = list(range(n))
        # 由于是按秩(连通块大小)合并,因此需要用数组记录每个连通块的大小
        self.size = [1] * n
        # 剩余连通块(集合)个数,初始数量为n
        self.cc = n

    # 扁平化
    def find(self, i):
        if i != self.father[i]:
            self.father[i] = self.find(self.father[i])
        return self.father[i]

    # 按秩合并(小集合挂大集合)
    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx == fy:
            return False
        if self.size[fx] >= self.size[fy]:
            self.size[fx] += self.size[fy]
            self.father[fy] = fx
        else:
            self.size[fy] += self.size[fx]
            self.father[fx] = fy
        self.cc -= 1
        return True

    # 将并查集重置成初始状态
    def reset(self):
        n = len(self.father)
        self.father = list(range(n))
        self.size = [1] * n
        self.cc = n

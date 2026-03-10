# 按秩合并,带标签和返回值的并查集,存储形式为数组,find函数用递归实现
class UnionFind:
    def __init__(self, n):
        self.father = list(range(n))
        self.size = [1] * n
        self.cc = n

    def find(self, i):
        if i != self.father[i]:
            self.father[i] = self.find(self.father[i])
        return self.father[i]

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

    def reset(self):
        n = len(self.father)
        self.father = list(range(n))
        self.size = [1] * n
        self.cc = n

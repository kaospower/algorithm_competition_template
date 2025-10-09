# 带标签和返回值的并查集,存储形式为数组
class UnionFind:
    def __init__(self, n):
        self.father = list(range(n))
        self.cc = n

    def find(self, i):
        if i != self.father[i]:
            self.father[i] = self.find(self.father[i])
        return self.father[i]

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx == fy:
            return False
        self.father[fx] = fy
        self.cc -= 1
        return True

    def reset(self):
        n = len(self.father)
        self.father = list(range(n))
        self.cc = n

    def separate(self, x, y):
        self.father[x], self.father[y] = x, y
        self.cc += 1

# 带标签和返回值的并查集,存储形式为数组
class UnionFind:
    def __init__(self, n):
        self.father = list(range(n))
        # 剩余连通块(集合)个数,初始数量为n
        self.cc = n

    def find(self, i):
        if i != self.father[i]:
            self.father[i] = self.find(self.father[i])
        return self.father[i]

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        # x,y在同一个集合,无需合并,返回False
        if fx == fy:
            return False
        # x,y不在同一个集合,需要合并,返回True,同时连通块数量-1
        self.father[fx] = fy
        self.cc -= 1
        return True

    # 将并查集重置成初始状态
    def reset(self):
        n = len(self.father)
        self.father = list(range(n))
        self.cc = n

    # 撤销union操作
    def separate(self, x, y):
        self.father[x], self.father[y] = x, y
        self.cc += 1

#撤销操作separate的具体用法,见模版题2092(https://leetcode.cn/problems/find-all-people-with-secret/)
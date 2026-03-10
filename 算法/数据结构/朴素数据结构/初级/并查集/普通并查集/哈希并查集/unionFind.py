#哈希并查集(用于处理字符串)
#哈希并查集初始化方法见下面示例
class UnionFind:
    def __init__(self):
        self.father = {}

    def find(self, i):
        if i != self.father[i]:
            self.father[i] = self.find(self.father[i])
        return self.father[i]

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx == fy:
            return False
        self.father[fx] = fy
        return True


# 哈希并查集初始化方法,可以边插入边合并,不过需要判断key是否在哈希表中
similarPairs=[]
u = UnionFind()
for x, y in similarPairs:
    if x not in u.father:
        u.father[x] = x
    if y not in u.father:
        u.father[y] = y
    u.union(x, y)
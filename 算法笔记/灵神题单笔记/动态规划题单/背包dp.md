# 1.01背包

# 2.完全背包

# 3.多重背包

# 4.分组背包

# 5.有依赖的背包(树形背包)

```c++
//3562(https://leetcode.cn/problems/maximum-profit-from-trading-stocks-with-discounts/)
//树形背包/有依赖的背包,至多写法
class Solution {
public:
    int maxProfit(int n, vector<int>& present, vector<int>& future, vector<vector<int>>& hierarchy, int budget) {
        vector<vector<int>> g(n);
        for (auto& e : hierarchy) {
            g[e[0] - 1].push_back(e[1] - 1);
        }

        auto dfs = [&](this auto&& dfs, int x) -> vector<array<int, 2>> {
            // 计算从 x 的所有儿子子树 y 中，能得到的最大利润之和
            vector<array<int, 2>> sub_f(budget + 1);
            for (int y : g[x]) {
                auto fy = dfs(y);
                for (int j = budget; j >= 0; j--) {
                    // 枚举子树 y 的预算为 jy
                    // 当作一个体积为 jy，价值为 fy[jy][k] 的物品
                    for (int jy = 0; jy <= j; jy++) {
                        for (int k = 0; k < 2; k++) {
                            sub_f[j][k] = max(sub_f[j][k], sub_f[j - jy][k] + fy[jy][k]);
                        }
                    }
                }
            }

            //然后考虑x是否购买
            vector<array<int, 2>> f(budget + 1);
            for (int j = 0; j <= budget; j++) {
                for (int k = 0; k < 2; k++) {
                    int cost = present[x] / (k + 1);
                    if (j >= cost) {
                        // 不买 x，转移来源是 sub_f[j][0]
                        // 买 x，转移来源为 sub_f[j-cost][1]，因为对于子树来说，父节点一定买
                        f[j][k] = max(sub_f[j][0], sub_f[j - cost][1] + future[x] - cost);
                    } else { // 只能不买 x
                        f[j][k] = sub_f[j][0];
                    }
                }
            }
            return f;
        };

        return dfs(0)[budget][0];
    }
};
```


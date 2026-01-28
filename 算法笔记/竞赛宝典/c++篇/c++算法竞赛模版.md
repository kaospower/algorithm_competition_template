# 1.快速幂,阶乘,逆元

**C++预处理要用auto init=[]{}();**

```c++
const int N=1000,mod=1e9+7;
using ll=long long;

ll fact[N],infact[N];
ll qmi(ll a,int k,int p){
    ll res=1;
    for (;k;k>>=1){
        if (k&1) res=res*a%p;
        a=a*a%p;
    }
    return res;
}
auto init=[]{
    fact[0]=1;
    infact[0]=1;
    for (int i=1;i<N;i++){
        fact[i]=fact[i-1]*i%mod;
        infact[i]=infact[i-1]*qmi(i,mod-2,mod)%mod;
    }
    return 0;
}();
```

# 2.记忆化搜索

```c++
//初始化n*n*n*n四维数组,默认值为-1
vector memo(n,vector(n+1,vector(n+1,vector<int>(n+1,-1))));
//固定写法,使得记忆化搜索不用带着缓存表参数,a,b,c,d为实际参数
//返回值和缓存表数值类型根据需要可换成long long
auto f=[&](this auto&& f,int a,int b,int c, int d)->int{
    int& res=memo[a][b][c][d];//添加引用使得修改res同时写入缓存表
    //如果res之前被修改过,直接返回
    if (res!=-1){
        return res;
    }
    /*
    *函数内容
    */
    return res;
};
```

另一种写法:见3562题

```c++
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



# 3.埃氏筛

```c++
//写在class上面
const int N=100;
vector<bool>p(N+1,true);
auto init=[]{
    p[0]=p[1]=false;
    for (int i=2;i<=sqrt(N);i++){
        if (p[i]){
            for (int j=i*i;j<=N;j+=i){
                p[j]=false;
            }
        }
    }
    return 0;
}();
```

# 4.单调队列

**这里给出的是单调递减队列**

```c++
deque<int>q;
vector<int>ans;
for (int r=0;r<nums.size();r++){
    while (!q.empty() && nums[r]>=nums[q.back()]){
        q.pop_back();
    }
    q.push_back(r);
    if (q.front()<=r-k){
        q.pop_front();
    }
    if (r>=k-1){
        ans.push_back(nums[q.front()]);
    }
}
```


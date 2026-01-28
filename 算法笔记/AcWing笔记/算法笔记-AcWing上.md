# 背包九讲

注意当空间优化成1维后,只有完全背包和多重背包单调队列优化体积是从小到大循环的,其余所有背包问题都是从大到小循环的

```c++
for 物品
	for 体积
		for 决策
```



## 1.01背包问题
每件物品只能选一次

```c++
/*
f[i][j]:前i个物品当前使用的总体积不超过j的情况下,所能获得的最大价值
result=max{f[n][0~v]}
1.不选第i个物品,f[i][j]=f[i-1][j];
2.选第i个物品,f[i][j]=f[i-1][j-v[i]]+val[i];
f[i][j]=max{1,2}
*/
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=1010;

int n,m;
int f[N][N]; //c++默认初始化成0
int v[N],w[N]; //v代表体积,w代表价值
int main()
{
    cin>>n>>m;
    for (int i=1;i<=n;i++) cin>>v[i]>>w[i];
    
    for (int i=1;i<=n;i++)
        for (int j=0;j<=m;j++)
        {
            f[i][j]=f[i-1][j];
            if (j>=v[i]) f[i][j]=max(f[i-1][j],f[i-1][j-v[i]]+w[i]);
        }
    cout<<f[n][m]<<endl;
    return 0;
}
```



```c++
//空间压缩
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=1010;

int n,m;
//如果把所有f[i]都初始化成0,就是<=j的最大价值
//如果把f[0]初始化成0,其余初始化成负无穷,就是恰好=j的最大价值
int f[N];
int v[N],w[N];
int main()
{
    cin>>n>>m;
    for (int i=1;i<=n;i++) cin>>v[i]>>w[i];
    
    for (int i=1;i<=n;i++)
        //从大到小枚举
        for (int j=m;j>=v[i];j--)
        	f[j]=max(f[j],f[j-v[i]]+w[i]);
    cout<<f[m]<<endl;
    return 0;
}
```

## 2.完全背包问题
每件物品可以选无限次

**注意只有完全背包体积是从小到大枚举,其他背包问题都是从大到小枚举**

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=1010;

int n,m;
int f[N];
int v[N],w[N];
int main()
{
    cin>>n>>m;
    for (int i=1;i<=n;i++) cin>>v[i]>>w[i];
    
    for (int i=1;i<=n;i++)
        //从小到大枚举
        for (int j=v[i];j<=m;j++)
        	f[j]=max(f[j],f[j-v[i]]+w[i]);
    cout<<f[m]<<endl;
    return 0;
}
```



## 3.多重背包问题
每件物品选的次数是不同且有限次

```c++
/*
f[i]总体积是i的情况下,最大价值是多少
两种初始化方法
1.f[i]=0;
f[m]
2.f[0]=0,f[i]=-INF,i!=0;
max{f[0...m]}
*/
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=110;

int n,m;
int f[N];

int main()
{
    cin>>n>>m;
    for (int i=0;i<n;i++)
    {
        int v,w,s;
        cin>>v>>w>>s;
        for (int j=m;j>=0;j--)
            for (int k=1;k<=s && k*v<=j;k++)
                f[j]=max(f[j],f[j-k*v]+k*w);
    }
    cout<<f[m]<<endl;
    return 0;
}
```



```c++
//多重背包二进制分组优化
/*
将多重背包转化成01背包,复制s份,变成01背包问题
7,
*/
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

const int N=2010;
int n,m;
int f[N];

struct Good
{
    int v,w;
};

int main()
{	
    vector<Good> goods;
    cin>>n>>m;
    for (int i=0;i<n;i++)
    {
        int v,w,s;
        cin>>v>>w>>s;
        for (int k=1;k<=s;k*=2)
        {
            s-=k;
            goods.push_back({v*k,w*k});
        }
        if (s>0) goods.push_back({v*s,w*s});
    }
    for (auto good:goods)
        for (int j=m;j>=good.v;j--)
            f[j]=max(f[j],f[j-good.v]+good.w);
    cout<<f[m]<<endl;
    return 0;
}
```



```c++
//单调队列优化+空间压缩(滚动数组实现)
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;
const int N=20010;
int n,m;
int f[N],g[N],q[N];
int main()
{
    cin>>n>>m;
    for (int i=0;i<n;i++)
    {
        int v,w,s;
        cin>>v>>w>>s;
        memcpy(g,f,sizeof f);
        for (int j=0;j<v;j++)
        {
            //单调队列
            int hh=0,tt=-1;
            for (int k=j;k<=m;k+=v)
            {
                if (hh<=tt && q[hh]<k-s*v) hh++;
                if (hh<=tt) f[k]=max(f[k],g[q[hh]]+(k-q[hh])/v*w);
                while (hh<=tt && g[q[tt]]-(q[tt]-j)/v*w<=g[k]-(k-j)/v*w) tt--;
                q[++tt]=k;
            }
        }
    }
    cout<<f[m]<<endl;
    return 0;
}
```

## 4.混合背包问题
把一些背包问题混在一起

```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;
const int N=1010;

int n,m;
int f[N];

struct Thing
{
    int kind;
    int v,w;
};
vector<Thing> things;
int main()
{
    cin>>n>>m;
    //输入时分01背包,完全背包,多重背包讨论,同时将多重背包利用二进制分组转化成01背包
    for (int i=0;i<n;i++)
    {
        int v,w,s;
        cin>>v>>w>>s;
        if (s<0) things.push_back({-1,v,w});
        else if (s==0) things.push_back({0,v,w});
        else
        {
            for (int k=1;k<=s;k*=2){
                s-=k;
                things.push_back({-1,v*k,w*k});
            }
            if (s>0) things.push_back({-1,v*s,w*s});
        }
    }
    //分01背包和完全背包两种情况求解,多重背包在输入时已经转化成了01背包
    for (auto thing:things)
    {
        if (thing.kind<0){
            for (int j=m;j>=thing.v;j--) f[j]=max(f[j],f[j-thing.v]+thing.w);
        }
        else
        {
            for (int j=thing.v;j<=m;j++) f[j]=max(f[j],f[j-thing.v]+thing.w);
        }
    }
    cout<<f[m]<<endl;
    return 0;
}
```

## 5.二维费用的背包问题
有两个维度的限制

```c++
#include <iostream>
#include <algorithm>

using namespace std;

const int N=110;

int n,v,m;
int f[N][N];

int main()
{
    cin>>n>>v>>m;
    for (int i=0;i<n;i++)
    {
        int a,b,c;
        cin>>a>>b>>c;
        for (int j=v;j>=a;j--)
            for (int k=m;k>=b;k--)
                f[j][k]=max(f[j][k],f[j-a][k-b]+c);
                
    }
    cout<<f[v][m]<<endl;
    return 0;
}
```

## 6.分组背包问题
有N组物品和一个容量是V的背包.每组物品有若干个,同一组内的物品最多只能选一个.每件物品的体积是$v_{ij}$,价值是

$w_{ij}$,其中i是组号,j是组内编号.求解将哪些物品装入背包,可使物品总体积不超过背包容量,且总价值最大.

```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;
const int N=110;
int n,m;
int f[N],v[N],w[N]; //f是dp数组,v是体积,w是价值

int main()
{
    cin>>n>>m;
    //枚举物品组
    for (int i=0;i<n;i++)
    {
        int s;
        cin>>s;
        for (int j=0;j<s;j++) cin>>v[j]>>w[j];
        //倒序枚举体积
        for (int j=m;j>=0;j--)
            //枚举组内物品
            for (int k=0;k<s;k++)
                if (j>=v[k])
                	f[j]=max(f[j],f[j-v[k]]+w[k]);
    }
    cout<<f[m]<<endl;
    return 0;
}
```

## 7.背包问题求方案数

```c++
//f[i][j]=max(f(i-1,j),f(i-1,j-vi)+wi),此题f[i][j]为了方便表示恰好等于v而不是<=v
//g[i][j]:f[i][j]取最大值的方案数

#include <cstring>
#include <iostream>
using namespace std;

const int N=1010,mod=1e9+7;

int n,m;
int f[N],g[N]; //g[j]保存体积恰好是j的方案数

int main()
{
    cin>>n>>m;
    
    memset(f,-0x3f,sizeof f); //将f[j]初始化成负无穷,j!=0
    f[0]=0;
    g[0]=1;
    for (int i=0;i<n;i++)
    {
        int v,w;
        cin>>v>>w;
        for (int j=m;j>=v;j--)
        {
            int maxv=max(f[j],f[j-v]+w);
            int cnt=0;
            if (maxv==f[j]) cnt+=g[j];
            if (maxv==f[j-v]+w) cnt+=g[j-v];
            g[j]=cnt%mod;
            f[j]=maxv;
        
        }
    }
    int res=0;
    for (int i=0;i<=m;i++) res=max(res,f[i]);
	int cnt=0;
    for (int i=0;i<=m;i++)
        if (res==f[i])
            cnt=(cnt+g[i])%mod;
    cout<<cnt<<endl;
    return 0;
}
```

## 8.背包问题求具体方案

```c++
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;
const int N=1010;

int n,m;
int v[N],w[N],f[N][N];

int main()
{
    cin>>n>>m;
    for (int i=1;i<=n;i++) cin>>v[i]>>w[i];
    //从后往前枚举
    for (int i=n;i>=1;i--)
        for (int j=0;j<=m;j++)
        {
            f[i][j]=f[i+1][j];
            if (j>=v[i]) f[i][j]=max(f[i][j],f[i+1][j-v[i]]+w[i]);
        }
    int vol=m;
    for (int i=1;i<=n;i++)
        if (vol>=v[i] && f[i][vol]==f[i+1][vol-v[i]]+w[i])
        {
            cout<<i<<' ';
            vol-=v[i];
        }
    return 0;
}
```

##  9.树上背包(有依赖的背包)

有N个物品和一个容量是V的背包.物品之间具有依赖关系,且依赖关系组成一棵树的形状.如果选择一个物品,则必须选择它的父节点.求解将哪些物品装入背包,可使物品总体积不超过背包容量,且总价值最大.



由于可以添加虚拟节点使得依赖关系变成一棵树,因此有依赖的背包问题也被称为树上背包问题

树上背包=树形dp+背包问题

本题对于节点u,其所有子树对应的最大价值可以看成物品组,相当于从物品组里选一个物品,因此可以转化成分组背包问题

```c++
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;
const int N=110;
int n,m;
int v[N],w[N];
int h[N],e[N],ne[N],idx;
int f[N][N]; //f[i][j]表示选节点i,体积最多是j的情况下以i为根的整棵子树最大收益

//邻接表基本操作
void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}
void dfs(int u)
{
    for (int i=h[u];~i;i=ne[i]) //遍历所有子节点,~i等价于i!=-1
    {
        int son=e[i];
        dfs(e[i]);
        //分组背包,必须选当前物品u(根节点),先给根节点留一个位置,将根节点体积抠出来
        for (int j=m-v[u];j>=0;j--) //循环体积
            for (int k=0;k<=j;k++) //循环决策,0～j看成分组内j+1种体积的物品
                f[u][j]=max(f[u][j],f[u][j-k]+f[son][k]);
    }
    //将物品u加进去
    for (int i=m;i>=v[u];i--) f[u][i]=f[u][i-v[u]]+w[u];
    //如果根节点选不了,即体积j<根节点体积,那么赋成0
    for (int i=0;i<v[u];i++) f[u][i]=0;
}
int main()
{
    cin>>n>>m;
    memset(h,-1,sizeof h);
    int root;
    for (int i=1;i<=n;i++)
    {
        int p;
        cin>>v[i]>>w[i]>>p;
        if (p==-1) root=i;
        else add(p,i);
        
    }
    dfs(root);
    cout<<f[root][m]<<endl;
    return 0;
}
```
# 排序算法

## 1.冒泡排序

**时间复杂度O(n^2),稳定**

## 2.插入排序

**时间复杂度O(n^2),稳定**

## 3.选择排序

**时间复杂度O(n^2),不稳定**

## 4.归并排序

**时间复杂度O(nlogn),稳定**

## 5.堆排序

**时间复杂度O(nlogn),不稳定**

## 6.快速排序

**时间复杂度O(nlogn),不稳定**

## 7.基数排序

**时间复杂度O(n),稳定**

## 8.希尔排序

**不稳定**

## 9.桶排序

**时间复杂度O(n)**

## 10.诱导排序

**基于桶排序,常用于SA-IS算法求后缀数组**

## 11.猴子排序

**用于揭示随机**

# ACWing基础课:手写堆



# ACWing基础课:基础算法

```c++
//快速排序思想
/*
基于分治
1.确定分界点:取左边界/取中间点/取右边界/随机
2.调整区间:保证<=x的数都在左边,>=x的数都在右边
3.递归处理左右两段
*/
```



```c++
//快排模版:双指针实现调整区间
#include<iostream>
using namespace std;
const int N=1e6+10;
int n;
int q[N];
void quick_sort(int q[],int l,int r)
{
    if (l>=r) return;
    int x=q[l],i=l-1,j=r+1 //两个指针分别指向边界两侧
    while (i<j)
    {
    	do i++;while (q[i]<x);
        do j--;while (q[j]>x);
        if (i<j) swap(q[i],q[j]);
    }
    quick_sort(q,l,j);
    quick_sort(q,j+1,r);
        
}
int main()
{
    scanf("%d",&n);
    for (int i=0;i<n;i++) scanf("%d",&q[i]);
    quick_sort(q,0,n-1);
    for (int i=0;i<n;i++) printf("%d",q[i]);
    return 0;
}
```

# ACWing基础课:前缀和

## 一维前缀和:

前缀和公式,下标必须从1开始

$原数组:a_1,a_2,a_3,...,a_n$

$前缀和: s_i=a_1+a_2+...+a_i,s_0=0$

```c++
for (int i=1;i<=n;i++) s[i]=s[i-1]+a[i]; //前缀和的初始化
ans=s[r]-s[l-1]; //区间和的计算
```

## 二维前缀和:

用来求子矩阵的和

```c++
for (int i=1;i<=n;i++)
    for (int j=1;j<=m;j++)
        s[i][j]=s[i-1][j]+s[i][j-1]-s[i-1][j-1]+a[i][j];//二维前缀和的初始化
ans=s[x2][y2]-s[x1-1][y2]-s[x2][y1-1]+s[x1-1][y1-1];//二维区间和的计算,即求矩形面积
```

# ACWing基础课:离散化

```txt
特指整数离散化
去重+映射
```



```c++
//去重
vector<int>::iterator unique(vector<int> &a)
{
    int j=0;
    for (int i=0;i<a.size();i++)
        if (!i||a[i]!=a[i-1])
            a[j++]=a[i]
    //a[0]~a[j-1] 所有a中不重复的数
    return a.begin()+j;
}
```



```c++
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef pair<int,int> PII;
const int N=300010;

int n,m;
int a[N],s[N];

vector<int> alls;
vector<PII> add,query;

int find(int x)
{
    int l=0,r=alls.size()-1;
    while (l<r)
    {
        int mid=l+r>>1;
        if (alls[mid]>=x) r=mid;
        else l=mid+1;
    }
    return r+1
}
int main()
{
    cin>>n>>m;
    for (int i=0;i<n;i++)
    {
        int x,c;
        cin>>x>>c;
        add.push_back({a,c});
        
        alls.push_back(x);
    }
    for (int i=0;i<m;i++)
    {
        int l,r;
        cin>>l>>r;
        query.push_back(l,r);
        
        alls.push_back(l);
        alls.push_back(r);
    }
    //去重
    sort(alls.begin(),alls.end());
    alls.erase(unique(alls.begin(),alls.end()),alls.end());
    for (auto item:add)
    {
        int x=find(item.first);
        a[x]+=item.second;
    }
    for (int i=1;i<=alls.size();i++) s[i]=s[i-1]+a[i];
    
    for (auto item:query)
    {
        int l=find(item.first),r=find(item.second);
        cout<<s[r]-s[l-1]<<endl;
    }
    return 0;
}
```

# ACWing基础课:区间合并

```c++
//将有交集的区间合并
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

typedef pair<int,int>PII;

const int N=100010;

int n;
vector<PII> seg;

void merge(vector<PII> &segs)
{
    vector<PII> res;
    sort(segs.begin(),segs.end());
    
    int st=-2e9,ed=-2e9;
    for (auto seg:segs)
        if (ed<seg.first){
            if (st!=-2e9) res.push_back({st,ed});
            st=seg.first,ed=seg.second;    
        }
    	else ed=max(ed,seg.second);
    if (st!=-2e9) res.push_back({st,ed});
    segs=res;
}
int main()
{
    cin>>n;
    for (int i=0;i<n;i++)
    {
        int l,r;
        cin>>l,r;
        segs.push_back({l,r});
    }
    merge(segs);
    cout<<segs.size()<<endl;
    return 0;
}
```

# ACWing基础课:链表和邻接表

```txt
用数组模拟单链表,如邻接表(链式前向星)

邻接表:存储树和图(n个链表)
双链表:优化某些问题
```



```c++
//用数组模拟单链表(静态链表),这种写法比指针+结构体写法(动态链表)更快,省去了new操作
#include <iostream>

using namespace std;

const int N=10010;

//head表示头节点的下标
//e[i]表示节点i的值,ne[i]表示节点i的下一个点
//idx表示当前已经用到了哪个点
//空节点下标用-1表示
int head,e[N],ne[N],idx;

//初始化
void init()
{
    head=-1;//head初始化成-1,表示链表为空
    idx=0;//下标从0开始分配
}

//将x插到头节点
void add_to_head(int x)
{
    e[idx]=x,ne[idx]=head,head=idx++;
}
//将x插到下标是k的点后面
void add(x,k)
{
    e[idx]=x,ne[idx]=ne[k],ne[k]=idx++;
}

//将下标是k的点后面的点删掉
void remove(int k)
{
    ne[k]=ne[ne[k]]
}
int main()
{
    int m;
    cin>>m;
    init();
    while (m--)
    {
        int k,x;
        char op;
        cin>>op;
        if (op=='H')
        {
            cin>>x;
            add_to_head(x);
        }
        else if (op=='D')
        {
            cin>>k;
            if (!k) head=ne[head] //删除头节点
            remove(k-1);
        }
        else
        {
            cin>>k>>x;
            add(k-1,x);
        }
        
    }
    for (int i=head;i!=-1;i=ne[i]) cout<<e[i]<<' ';
    cout<endl;
    return 0;
}
```



```c++
//邻接表(链式前向星/n个链表)dfs
//求树的重心
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N=100010,M=N*2;

//h表示头节点数组,h[a]这个链表存储父节点是a的所有点的地址(虚拟下标idx,代表第几条边)
//也可以理解成h[a]表示所有以a为起点的边的序号(idx),在知道这个序号后,就可以通过e[idx]来访问边的终点
//e[idx]表示通过虚拟下标idx(边)指向的实际点的下标,即e保存边的终点
int h[N],e[M],ne[M],idx; 
bool st[N];//记录点是否遍历过

int ans=N;
void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}

//返回以u为根的子树中点的数量
void dfs(int u)
{
    st[u]=true;
    int sum=1,res=0;
    //也可以写成 for (int i=h[u];~i;i=ne[i]),因为-1取反是0,只有i!=-1时,~i才为真
    for (int i=h[u];i!=-1;i=ne[i])
    {
        int j=e[i];
        if (!st[j]){
            int s=dfs(j);
            res=max(res,s);
            sum+=s;
        } 
    }
    res=max(res,n-sum);
    ans=min(ans,res);
    return sum;
}
int main()
{
    cin>>n;
    memset(h,-1,sizeof h);//初始化
    for (int i=0;i<n-1;i++)
    {
        int a,b;
        cin>>a>>b;
        add(a,b),add(b,a);
    }
    dfs(1);
    cout<<ans<<endl;
    return 0;
}
```



```c++
//邻接表bfs
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;
const int N=100010;
int n,m;
int h[N],e[N],ne[N],idx;
int d[N],q[N];

void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}
int bfs()
{
    int hh=0,tt=0;//队头,队尾
    q[0]=1;
    memset(d,-1,sizeof d); //d表示距离
    d[1]=0;
    while (hh<=tt)
    {
        int t=q[hh++];//队头
        for (int i=h[t];~i;i=ne[i]){
            int j=e[i];
            if (d[j]==-1){
                d[j]=d[t]+1;
                q[++tt]=j;
            }
        }
    }
    return d[n];
}
int main(){
    cin>>n>>m;
    memset(h,-1,sizof h);
    for (int i=0;i<m;i++)
    {
        int a,b;
        cin>>a>>b;
        add(a,b);
    }
    cout<<bfs()<<endl;
    return 0;
}

```

# ACWing基础课:拓扑排序

```txt
有向图才有拓扑序列
有环图不存在拓扑序
有向无环图(DAG)一定存在拓扑序
因此,有向无环图也被称为拓扑图
点的入度:有多少条边指向自己
点的出度:有几条边出去

入度为0的点可以作为起点
1.将所有入度为0的点入队
2.从队列弹出一个点t,枚举t的所有出边t->j
3.删掉t->j,d[j]--,即d的入度减1
4.如果d[j]==0,将j加入队列

一个有向无环图一定至少存在一个入度为0的点
拓扑排序可以判断图是否有环

注意有向无环图拓扑排序答案不为1
```



```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=100010;

int n,m;
int h[N],ne[N],e[N],idx;
int q[N],d[N];//q是队列,d存储点的入度

void add(int a,int b){
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}

bool topsort()
{
    int hh=0,tt=-1;
    for (int i=1;i<=n;i++)
        if (!d[i])
            q[++tt]=i;//将所有入度为0的点插入队列中
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int j=e[i];
            d[j]--;
            if (d[j]==0) q[++tt]=j;
        }
    }
    return tt==n-1; //判断队列是否进入过n个点,如果不是说明有环,返回false
}
int main()
{
    cin>>n>>m;
    memeset(h,-1,sizof h);
    for (int i=0;i<m;i++)
    {
        int a,b;
        cin>>a>>b;
        add(a,b);
        d[b]++;
    }
    if (topsort())
    {
        for (int i=0;i<n;i++) printf("%d",q[i]);
        puts("");
    }
    else puts("-1");
    return 0;
}
```



# ACWing基础课:Trier(字典树/前缀树)

```c++
//高效地存储和查找字符串集合的数据结构,把所有字符串结尾打上标记
//模版题:Trie字符串统计
#include <iostream>
using namespace std;
const int =100010;
//son:存储每个点的所有儿子,cnt:以当前点结尾的单词有多少个,idx和单链表中的idx一样;
//下标是0的点,既是根节点,又是空节点
int son[N][26],cnt[N],idx;
char str[N];

//插入
void insert(char str[])
{
    int p=0;
    for (int i=0;str[i];i++)
    {
        int u=str[i]-'a';
        if (!son[p][u]) son[p][u]=++idx;
        p=son[p][u];
    }
    cnt[p]++;
    
}

//查询
int query(char str[])
{
    int p=0;
    for (int i=0;str[i];i++)
    {
        int u=str[i]-'a';
        if (!son[p][u]) return 0;
        p=son[p][u];
    }
    return cnt[p];
}

int main()
{
    int n;
    scanf("%d",&n);
    while (n--)
    {
        char op[2];
        scanf("%s%s",op,str);
        if (op[0]=='I') insert(str);
        else printf("%d\n",query(str));
    }
    return 0;
}
```

# ACWing基础课:手写堆(支持堆中任意位置元素的删除)

```c++
```



# ACWing基础课:KMP

```c++
#include <iostream>
using namespace std;
const int N=10010,M=100010;
int n,m;
char p[N],s[M];
int ne[N];
int main(){
    cin>>n>>p+1>>m>>s+1;
    //求解next的过程
    for (int i=2,j=0;i<=n;i++)
    {
        while (j&&p[i]!=p[j+1]) j=ne[j];
        if (p[i]==p[j+1]) j++;
        ne[i]=j;
    //kmp匹配过程
    for (int i=1,j=0;i<=m;i++)
    {
        while (j&&s[i]!=p[j+1]) j=ne[j];
        if (s[i]==p[j+1]) j++;
        if (j==n)
        {
            printf("%d",i-n);
            j=ne[j];
        }
    }
    return -1;
}
```



```python
#KMP模版
def KMP(s,p):
    m,n=len(s),len(p)
    ne=[0]*(n+1) #next数组,ne[i]表示0...i范围(不包含i)前缀和后缀的最大匹配长度
    #TODO:求解next数组的过程
    j=0
    for i in range(1,n):
        #s[0...i]范围和p[0...j]范围匹配,如果发现s[i]和p[j]不匹配
        #将s[i-1]和p[ne[j]]匹配,再比较s[i]和p[j]
        while j>0 and p[i]!=p[j]:
            j=ne[j]
        if p[i]==p[j]:
            j+=1
        ne[i+1]=j
    #TODO:KMP匹配过程
    j=0
    for i in range(m):
        while j>0 and s[i]!=p[j]:
            j=ne[j]
        if s[i]==p[j]:
            j+=1
        if j==n:
            return i-n+1
    return -1
```



# AcWing基础课:最短路

无向图是特殊的有向图,以下所有算法对是有向图还是无向图没有要求,可以用有向图的算法解决无向图的问题
$$
\begin{equation*}
最短路
\begin{cases}
	\begin{aligned}
	&单源最短路:\\
	&(求源点到其他所有点的最短距离)\\
	\end{aligned}
  	\begin{cases}
    	所有边权都是正数
    	\begin{cases}
    	\begin{aligned}
    	&朴素\bold{Dijkstra}算法,时间复杂度O(n^2),和边数没有关系\\
    	&(适用于稠密图,边数较多时,如m=n^2时,此时O(mlogn)=O(n^2logn)>O(n^2))\\
    	&其中n代表点数,m代表边数\\
    	&稠密图用邻接矩阵存\\
    	\\
    	\\
    	\end{aligned}\\
    	\begin{aligned}
    	&堆优化版的\bold{Dijkstra}算法,时间复杂度O(mlogn),注意严格写法是(m+n)logn,一般由于m>=n,所以简写成mlogn,手写堆实现是mlogn,用优先队列实现是mlogm,因为优先队列实现的堆存在冗余点\\
    	&(适用于稀疏图,边数较少时,如m=n时,此时O(mlogn)=O(nlogn)<O(n^2))\\
    	&稀疏图用邻接表存\\
    	\end{aligned}
    	\end{cases}\\
    	存在负权边
    	\begin{cases}
    	\begin{aligned}
    	&\bold{Bellman-Ford}算法,时间复杂度O(nm)\\
    	&(如果限制边数,就只能用Bellman-Ford算法)\\
    	\end{aligned}\\
    	\begin{aligned}
    	&\bold{SPFA},时间复杂度一般O(m),最坏O(nm),对Bellman-Ford的优化,效率一般比Bellman-Ford算法高\\
    	\end{aligned}
    	\end{cases}\\
    \end{cases}\\
    \\
    \begin{aligned}
	&多源汇最短路:\\
	&(任选两个点,求它们之间的最短距离,即有很多源点,其中源点为起点,汇点为终点)\\
	\end{aligned}
	\quad \bold{Floyd}算法,时间复杂度O(n^3)\\
\end{cases}
\end{equation*}
$$

```mysql
重要考察点:如何建图
```

**朴素Dijkstra算法**:

① 初始化距离:$dist[1]=0,dist[i]=+\infty$

② s:所有当前已经确定最短距离的点的集合

找到不在s中的距离最近的点t

将t加到s中去

用t更新其他点的距离,看dist[x]是否大于dist[t]+w,如果大于,就更新距离

循环n次可以确定每个点到源点的最短距离

```c++
//模版
#include<cstring>
#include<iostream>
#include<algorithm>

using namespace std;
const int N=510
int n,m;
int g[N][N]; //邻接矩阵
int dist[N]; //最短距离
bool st[N]; //每个点最短距离是否确定了

int dijkstra()
{
    memset(dist,0x3f,sizeof dist); //0x3f代表正无穷
    dist[1]=0;
    for (int i=0;i<n-1;i++){
    	int t=-1;
        for (int j=1;j<=n;j++)
            if (!st[j]&&(t==-1||dist[t]>dist[j]))
                t=j
        st[t]=true;
        for (int j=1;j<=n;j++)
            dist[j]=min(dist[j],dist[t]+g[t][j]);
    }
    if (dist[n]==0x3f3f3f3f) return -1;
    return dist[n];
}
int main(){
    //对于重边和自环,只保留距离最短的边
    scanf("%d%d",&n,&m);
    //初始化
    memset(g,0x3f,sizeof g);
    /*
    for (int i=1;i<=n;i++)
        for (int j=1;j<=n;j++)
            if (i==j) g[i][j]=0;
    		else g[i][j]=INF;
    */
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        g[a][b]=min(g[a][b],c);//只保留a和b之间长度最短的一条边
        
    }
    int t=dijkstra();
    printf("%d\n",t);
    return 0;
}
```

**堆优化Dijkstra算法**:

```c++
//堆优化Dijkstra算法
//模版
#include<cstring>
#include<iostream>
#include<algorithm>
#include<queue>

using namespace std;

typedef pair<int ,int>PII;//用pair存储节点编号
const int N=100010
int n,m;
int h[N],e[N],w[N],ne[N],idx;//邻接表,w存储权重
int dist[N]; //最短距离
bool st[N]; //每个点最短距离是否确定了

void add(int a,int b,int c)
{
    e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
}
int dijkstra()
{
    memset(dist,0x3f,sizeof dist); //0x3f代表正无穷
    dist[1]=0;
    priority_queue<PII,vector<PII>,greater<PII>>heap;//小根堆
    heap.push({0,1});
    while (heap.size())
    {
        auto t=heap.top();
        heap.pop();
        int ver=t.second,distance=t.first;
        if (st[ver]) continue; //如果发现冗余点,continue
        
        for (int i=h[ver];~i;i=ne[i])
        {
            int j=e[i];
            if (dist[j]>distance+w[i]){
                dist[j]=distance+w[i];
                heap.push({dist[j],j});
            }
        }
    }
    
    if (dist[n]==0x3f3f3f3f) return -1;
    return dist[n];
}
int main(){
    //对于重边和自环,只保留距离最短的边
    scanf("%d%d",&n,&m);
    //初始化
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c)
        
    }
    int t=dijkstra();
    printf("%d\n",t);
    return 0;
}
```

**Bellman-Ford算法**:

时间复杂度O(nm)

迭代n次,循环所有边

for n次

	for 所有边 a,b,w
	
		dist[b]=min(dist[b],dist[a]+w) 松弛操作

Bellman-Ford算法运行结束后,对于所有边都满足三角不等式dist[b]<=dist[a]+w

注意如果有负权回路,最短路就不一定存在了

Bellman-Ford第一重循环迭代k次表示不超过k条边最短路的距离

如果迭代了n次,说明有n条边,对应n+1个点,而我们只有n个点,说明存在负环

不过我们一般用效率更高的SPFA算法来寻找负环

```c++
//Bellman-Ford算法
//有边数限制的最短路

#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N=510,M=10010;

int n,m;
int dist[N],backup[N];

struct Edge
{
    int a,b,w;
}edges[M];

int bellman_ford()
{
    memeset(dist,0x3f,sizeof dist);
    dist[1]=0;
    for (int i=0;i<k;i++){
        memcpy(backup,dist,sizeof dist);//备份dist数组,只用上一次迭代的结果,防止发生串联
        for (int j=0;j<m;j++)
        {
            int a=edges[j].a,b=edges[j].b,w=edges[j].w;
            dist[b]=min(dist[b],backup[a]+w);
        }
    }
    if (dist[n]>0x3f3f3f3f/2) return -1;
    return dist[n];
}
int main()
{
    scanf("%d%d%d",&n,&m,&k);
    
    for (int i=0;i<m;i++)
    {
        int a,b,w;
        scanf("%d%d%d",&a,&b,&w);
        edges[i]={a,b,w};
    }
    int t=bellman_ford();
    
    if (t==-1) puts("impossible");
    else printf("%d\n",t);
    
    return 0;
}
```

**spfa算法**:

网格图容易卡spfa

spfa算法适用于无负环图

队列中存放待更新点

将起点放到队列中

取出队头t

更新t的所有出边t->b

将b加入队列

```c++
//spfa算法求最短路模版
#include<cstring>
#include<iostream>
#include<algorithm>
#include<queue>

using namespace std;

typedef pair<int ,int>PII;
const int N=100010
int n,m;
int h[N],e[N],w[N],ne[N],idx;
int dist[N]; 
bool st[N];

void add(int a,int b,int c)
{
    e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
}
int spfa()
{
    memset(dist,0x3f,sizeof dist);
    dist[1]=0;
    queue<int>q;
    q.push(1);
    st[1]=true;//当前点是否在队列当中,防止存放重复点
    while (q.size())
    {
        int t=q.front();
        q.pop();
        st[t]=false;
        for (int i=h[t];~i;i=ne[i])
        {
            int j=e[i];
            if (dist[j]>dist[t]+w[i]){
                dist[j]=dist[t]+w[i];
                if (!st[j])
                {
                    q.push(j);
                    st[j]=true;
                }
            }
        }
    }
    if (dist[n]==0x3f3f3f3f) return -1;
    return dist[n];
}
int main(){
    //对于重边和自环,只保留距离最短的边
    scanf("%d%d",&n,&m);
    //初始化
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c)
        
    }
    int t=spfa();
    if (t==-1) puts("impossible");
    else printf("%d\n",t);
    return 0;
}
```



```c++
//spfa判负环模版
//dist[x] 1~x最短距离
//cnt[x] 当前最短路边的数量
//dist[x]=dist[t]+w[i],cnt[x]=cnt[t]+1
//如果cnt[x]>=n,说明存在负环

#include<cstring>
#include<iostream>
#include<algorithm>
#include<queue>

using namespace std;

typedef pair<int ,int>PII;
const int N=100010
int n,m;
int h[N],e[N],w[N],ne[N],idx;
int dist[N],cnt[N]; 
bool st[N];

void add(int a,int b,int c)
{
    e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
}
int spfa()
{
    queue<int>q;
    //一开始将所有点加入队列中
    for (int i=0;i<=n;i++){
        st[i]=true;
        q.push(i);
    }
    while (q.size())
    {
        int t=q.front();
        q.pop();
        st[t]=false;
        for (int i=h[t];~i;i=ne[i])
        {
            int j=e[i];
            if (dist[j]>dist[t]+w[i]){
                dist[j]=dist[t]+w[i];
                cnt[j]=cnt[t]+1;
                if (cnt[j]>=n) return true;
                if (!st[j])
                {
                    q.push(j);
                    st[j]=true;
                }
            }
        }
    }
    return false;
}
int main(){
    //对于重边和自环,只保留距离最短的边
    scanf("%d%d",&n,&m);
    //初始化
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c)
        
    }
    if (spfa()) puts("Yes");
    else puts("No");
    return 0;
}
```



**Floyd算法**:

$d[i][j]邻接矩阵存储所有边$

```c++
//循环之前d[i][j]表示边,循环结束之后d[i][j]表示最短路
//Floyd算法基于动态规划
//d[k,i,j]表示从i出发,只经过1...k这些中间点到j的最短距离,第一维可以优化掉
for (int k=1;i<=n;k++)
    for (int i=1;i<=n;i++)
        for (int j=1;j<=n;j++)
            d[i][j]=min(d[i][j],d[i][k]+d[k][j])
```



````c++
#include<cstring>
#include<iostream>
#include<algorithm>

//重边取最小,自环直接删掉
using namespace std;
const int N=210,INF=1e9;
int n,m,Q;
int d[N][N];//邻接矩阵

void floyd()
{
    for (int k=1;k<=n;k++)
        for (int i=1;i<=n;i++)
            for (int j=1;j<=n;j++)
                d[i][j]=min(d[i][j],d[i][k]+d[k][j]);
}
int main(){
	scanf("%d%d%d",&n,&m,&Q);
    for (int i=1;i<=n;i++)
        for (int j=1;j<=n;j++)
        	if (i==j) d[i][j]=0; //初始化,处理自环
    		else d[i][j]=INF;
    while (m--){
        int a,b,w;
        scanf("%d%d%d",&a,&b,&c);
        d[a][b]=min(d[a][b],w); //初始重边
        
    }
    floyd();
    while (Q--)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        if (d[a][b]>INF/2) puts("impossible");
        else printf("%d\n",d[a][b]);
    }
    return 0;
}
````

# AcWing基础课:二分图

**二分图定义**:如果一张无向图的N个节点(N>=2)可以分成A,B两个非空集合,其中$A\cap B=\varnothing$,并且在同一集合内的点之间都没有边相连,那么称这张无向图为一张二分图.A,B分别称为二分图的左部和右部.

**二分图判定**:一张图是二分图,当且仅当图中不存在奇环(长度为奇数的环).

**匹配**:任意两条边都没有公共端点的边的集合被称为图的一组匹配.

**最大匹配**:在二分图中,包含边数最多的一组匹配被称为二分图的最大匹配.

**增广路**(交错路):对于任意一组匹配S(S是一个边集),属于S的边被称为匹配边,不属于S的边被称为非匹配边.匹配边的端点被称为匹配点,其他节点被称为非匹配点.如果在二分图中存在一条连接两个非匹配点的路径path,使得非匹配边与匹配边在path上交替出现,那么称path是匹配S的增广路,也称交错路.

增广路性质:

-   1.长度len是奇数
-   2.路径上第1,3,5...,len条边是非匹配边,第2,4,6,...,len-1条边是匹配边.

因为以上性质,如果我们把路径上所有边的状态取反,原来的匹配边变成非匹配边,原来的非匹配边变成匹配的,那么得到的新的边集S'仍然是一组匹配,并且匹配边数增加了1.进一步可以得到推论:

**推论**:二分图的一组匹配S是最大匹配,当且仅当图中不存在S的增广路



**匈牙利算法**(增广路算法)

匈牙利算法用于计算二分图最大匹配

**步骤**

-   1.设$S=\varnothing$,即所有边都是非匹配边
-   2.寻找增广路path,把路径上所有边的匹配状态取反,得到一个更大的匹配S'
-   3.重复第二步,直至图中不存在增广路

该算法的关键在于如何找到一条增广路.匈牙利算法依次尝试给每一个左部节点x寻找一个匹配的右部节点y.右部点y能与左部点x匹配,需要满足以下两个条件之一:

-   1.y本身就是非匹配点.此时无向边(x,y)本身就是非匹配边,自己构成一条长度为1的增广路
-   2.y已经与左部点x'匹配,但从x'出发能找到另一个右部点y'与之匹配.此时路径x~y~x'~y'为一条增广路.

在实际的程序实现中,我们采用dfs的框架,递归地从x出发寻找增广路.若找到,则在深搜回溯时,正好把路径上的匹配状态取反。另外,可以用全局bool数组标记节点的访问情况,避免重复搜索.

匈牙利算法的正确性基于贪心策略,它的一个重要特点是:当一个节点称为匹配点后,至多因为找到增广路而更换匹配对象,但是绝不会再变回非匹配点

对于每个左部节点,寻找增广路最多遍历整张二分图一次,因此该算法时间复杂度为O(NM),N为顶点数,M为边数

```c++
//匈牙利算法模版
//下标从1开始,初始时所有点默认匹配0,代表未匹配状态
bool dfs(int x){
    for (int i=h[x];i;i=ne[i])
        int y=e[i];
        if (!visit[y]){
            visit[y]=true;
            if (match[y]==0 || dfs(match[y])){
                match[y]=x;
                return true;
            }
        }
    return false;
}
for (int i=1;i<=n;i++){
    memset(visit,false,sizeof(visit));
    if (dfs(i)) ans++;
}
```



```c++
//染色法判断二分图
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

constage int N=100010,M=200010;

int n,m;
int h[N],e[M],ne[M],idx;
int color[N];

void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}

bool dfs(int u,int c)
{
    color[u]=c;
    for (int i=h[u];i!=-1;i=ne[i])
    {
        int j=e[i];
        if (!color[j])
        {
            if (!dfs(j,3-c)) return false;
        }
        else if (color[j]==c) return false;
        
    }
}
int main()
{
    scanf("%d%d",&n,&m);
    memset(h,-1,sizof h);
    while (m--)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        add(a,b),add(b,a);
    }
    bool flag=true;
    for (int i=1;i<=n;i++)
        if (!color[i])
        {
            if (!dfs(i,1))
            {
                flag=false;
                break;
            }
        }
	if (flag) put("Yes");
    else puts("No");
    return 0;
}
```



```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N510,M=10010;

int n1,n2,m;
int h[N],e[M],ne[M],idx;
int match[N];
bool st[N];

void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}
bool find(int x)
{
    for (int i=h[x];i!=-1;i=ne[i])
    {
        int j=e[i];
        if (!st[j])
        {
            st[j]=true;
            if (match[j]==0||find(match[j]))
            {
                match[j]=x;
                return true;
            }
        }
    }
    return false;
}
int main()
{
    scanf("%d%d%d",&n1,&n2,&m);
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        add(a,b);
    }
    int res=0;
    for (int i=1;i<=n1;i++)
    {
        memset(st,false,sizeof st);
        if (find(i)) res++;
    }
    printf("%d\n",res);
    return 0;
}
```



```txt
有权二分图最大匹配问题(Maximum-Weight Bipartite Matching)
匈牙利算法:
时间复杂度O(n^3)
要求二分图左右两个子集元素数量必须相同都是n,即邻接矩阵必须是n x n的方阵
匈牙利算法解决最小权匹配问题,对于最大权匹配问题,将权重取反然后调用最小权匹配算法即可
```



**二分图带权匹配**:给定一张二分图,二分图的每条边都带有一个权值.求出该二分图的一组最大匹配,使得匹配边的权值总和最大.这个问题称为二分图的带权最大匹配,也称二分图最优匹配.注意,二分图带权最大匹配的前提是匹配数最大,然后再最大化匹配边的权值总和.

二分图带权匹配有两种解法:费用流和KM算法.KM算法程序实现简单,在稠密图上的效率一般高于费用流.不过,KM算法有很大的局限性,只能在满足带权最大匹配一定是完备匹配的图中正确求解.一般鼓励用费用流来计算二分图带权最大匹配.

在接下来的讨论中,我们设二分图左,右两部的节点数均为N.

**交错树**:在匈牙利算法中,如果从某个左部节点出发寻找匹配失败,那么在dfs过程中,所有访问过的节点,及为了访问这些节点而经过的边,共同构成一棵树.这棵树的根是一个左部节点,所有叶子节点也都是左部节点(因为最终匹配失败了),并且树上第1,3,5,...层的边都是非匹配边,第2,4,6,...层的边都是匹配边.因此,这棵树被称为交错树.

**顶标**:全称顶点标记值.在二分图中,给第i(1<=i<=N)个左部节点一个整数值$A_i$,给第j(1<=j<=N)个右部节点一个整数值$B_j$.同时,必须满足$\forall i,j,A_i+B_j>=w(i,j)$,其中w(i,j)表示第i个左部点与第j个右部点之间的边权(没有边时设为负无穷).这些整数值$A_i,B_j$称为节点的顶标.

**相等子图**:二分图中所有节点和满足$A_i+B_j=w(i,j)$的边构成的子图,称为二分图的相等子图.

**定理**:若相等子图中存在完备匹配,则这个完备匹配就是二分图的带权最大匹配.

**KM算法**:先在满足$\forall i,j,A_i+B_j>=w(i,j)$的前提下,给每个节点随意赋值一个顶标,然后采取适当的策略不断扩大相等子图的规模,直至相等子图存在完备匹配.对于一个相等子图,我们用匈牙利算法求它的最大匹配.若最大匹配不完备,则说明一定有一个左部节点匹配失败.该节点匹配失败的那次dfs形成了一颗交错树,记为T.

考虑匈牙利算法的流程,容易发现以下两条结论:

**结论**:

-   1.除了根节点以外,T中其他的左部点都是从右部点沿着匹配边访问到的,即在程序中调用了dfs(math[y]),其中y是一个右部节点.
-   2.T中所有的右部点都是从左部点沿着非匹配边访问到的

在寻找到增广路以前,我们不会改变已有匹配,所以一个右部点沿着匹配边能访问到的左部点是固定的.为了让匹配数增加,我们只能从第2条结论入手,考虑怎样能让左部点沿着非匹配边访问到更多的右部点.

假如我们把交错树T中的所有左部节点顶标$A_i(i\in T)$减少一个整数值$\Delta$,把T中所有右部节点顶标$B_j(j\in T)$增大一个整数值$\Delta$,节点的访问情况会有哪些变化,我们分两方面进行讨论:

-   1.右部点j沿着匹配边,递归访问i=match[j]的情形.对于一条匹配边,显然要么$i,j\in T$(被访问到),要么$i,j\notin T$(没被访问到).故$A_i+B_j$不变,匹配边仍然属于相等子图.
-   2.左部点i沿着非匹配边,访问右部点j,尝试与之匹配的情形.因为左部点的访问是被动的(被右部点沿着匹配边递归),所以只需考虑$i\in T$.
    -   (1)若$i,j\in T,则A_i+B_j$不变.即以前能从i访问到的点j,现在仍能访问.
    -   (2)若$i\in T,j\notin T,则A_i+B_j$减小,即以前能从i访问不到的点j,现在有可能访问到了.

为了保证顶标符合前提条件$\forall i,j,A_i+B_j>=w(i,j)$,我们就在所有$i\in T,j\notin T$的边(i,j)之中,找出最小的$A_i+B_j-w(i,j)$,作为$\Delta$的值.只要原图存在完备匹配,这样的边一定存在.上述方法既不会破坏前提条件,又能保证至少有一条新的边会加入相等子图,使交错树至少一个左部点能访问到的右部点增多.

不断重复以上过程,直到每一个左部点都匹配成功,就得到了相等子图的完备匹配,即原图的带权最大匹配.具体实现时,可以在dfs的过程中维护一个数组记录可能更新$\Delta$的值,以便快速求出新的$\Delta$.时间复杂度为$O(N^4)$,随机数据$O(N^3)$.

```c++
//KM算法模版
const int N=105;
int w[N][N];//边权
int la[N],lb[N];//左,右部点的顶标
bool va[N],vb[N];//访问标记:是否在交错树中
int match[N];//右部点匹配了哪一个左部点
int n,delta,upd[N];

bool dfs(int x){
    va[x]=true;//访问标记:x在交错树中
    for (int y=1;y<=n;y++)
        if (!vb[y])
            if (la[x]+lb[y]-w[x][y]==0){//相等子图
                vb[y]=true;//访问标记:y在交错树中
                if (match[y]==0 || dfs(match[y])){
                    match[y]=x;
                    return true;
                }
            }
    		else upd[y]=min(upd[y],la[x]+lb[y]-w[x][y]);
    return false;
}
int KM(){
    for (int i=1;i<=n;i++){
        la[i]=-(1<<30);//-inf
        lb[i]=0;
        for (int j=1;j<=n;j++)
            la[i]=max(la[i],w[i][j]);
    }
    for (int i=1;i<=n;i++)
        while (true){//直到左部点找到匹配
            memset(va,false,sizeof(va));
            memset(vb,false,sizeof(vb));
            delta=inf;
            for (int j=1;j<=n;j++) upd[j]=1e10; //inf
            if (dfs(i)) break;
            for (int j=1;j<=n;j++)
                if (!vb[j]) delta=min(delta,upd[j]);
            for (int j=1;j<=n;j++){
                if (va[j]) la[j]-=delta;
                if (vb[j]) lb[j]+=delta;
            }
        }
    int ans=0;
    for (int i=1;i<=n;i++) ans+=w[match[i]][i];
    return ans;
}
```



```python
#KM算法(dfs),最大带权二分图匹配
#力扣1947
#KM算法(dfs)
min=lambda x,y:x if x<y else y
def KM(w):
    n=len(w)
    #w:边权,la:左部点顶标,lb:右部点顶标,va,vb:访问标记:是否在交错树中,mt:右部点匹配了哪一个左部点
    la,lb,mt,upd,va,vb,=[0]+[max(w[i][j] for j in range(1,n)) for i in range(1,n)],[0]*n,[0]*n,[inf]*n,[False]*n,[False]*n
    def dfs(x):
        va[x]=True
        for y in range(1,n):
            if not vb[y]:
                if la[x]+lb[y]-w[x][y]==0:
                    vb[y]=True
                    if mt[y]==0 or dfs(mt[y]):
                        mt[y]=x
                        return True
                else:upd[y]=min(upd[y],la[x]+lb[y]-w[x][y])
        return False
    for i in range(1,n):
        while True:
            for j in range(1,n):
                va[j],vb[j],upd[j]=False,False,inf
            if dfs(i):break
            delta=inf      
            for j in range(1,n):
                if not vb[j]:delta=min(delta,upd[j])
            for j in range(1,n):
                if va[j]:la[j]-=delta
                if vb[j]:lb[j]+=delta
    return sum(w[mt[i]][i] for i in range(1,n))
class Solution:
    def maxCompatibilitySum(self, students: List[List[int]], mentors: List[List[int]]) -> int:
        n=len(students)+1
        w=[[0]*n for _ in range(n)] #邻接矩阵表示二分图,下标从1开始
        #预处理边权
        for i,x in enumerate(students):
            for j,y in enumerate(mentors):
                w[i+1][j+1]=sum(a==b for a,b in zip(x,y))
        return KM(w)
```



```python
#KM算法(bfs),最大带权二分图匹配
def KM(w):
    """
    vb:记录右侧顶点是否已在交错树中
    slack:记录每个右侧顶点与当前交错树的差距
    last:记录交错树中右侧顶点的前驱，用于回溯增广路径
    y:从哑结点开始构造交错树
    mt[0]=i:假设左侧顶点 i 与哑结点 0 匹配
    x:当前 y 对应的左侧顶点
    """
    n=len(w)
    mt,la,lb=[0]*n,[0]+[max(w[i][j] for j in range(1, n)) for i in range(1, n)],[0] * n
    # 对于每个真实左侧顶点寻找匹配
    for i in range(1,n):
        vb,slack,last,y,mt[0]=[False]*n,[inf]*n,[0]*n,0,i
        while True:
            vb[y],x,delta,nexty=True,mt[y],inf,0
            # 更新所有未加入交错树右侧顶点的 slack 值
            for j in range(1,n):
                if not vb[j]:
                    if (d:=la[x]+lb[j]-w[x][j])<slack[j]:slack[j],last[j]=d,y
                    if slack[j]<delta:delta,nexty=slack[j],j
            # 如果 delta > 0，则调整所有标号
            if delta>0:
                for j in range(n):
                    if vb[j]:
                        la[mt[j]]-=delta
                        lb[j]+=delta
                    else:slack[j]-=delta
            y=nexty
            # 找到未匹配的右侧顶点时退出循环
            if mt[y]==0:break
        # 倒推更新增广路
        while y>0:mt[y],y=mt[last[y]],last[y]
    return sum(w[mt[i]][i] for i in range(1,n))
class Solution:
    def maxCompatibilitySum(self, students, mentors):
        n=len(students)+1
        w=[[0]*n for _ in range(n)]
        # 预处理边权
        for i,x in enumerate(students):
            for j,y in enumerate(mentors):
                w[i+1][j+1]=sum(a==b for a,b in zip(x,y))
        return KM(w)

```



```python
#KM算法(dfs),最小带权二分图匹配
#在最大带权二分图模版匹配基础上,将边权取反,最后将结果取反即可
#力扣1879
min=lambda x,y:x if x<y else y
def KM(w):
    n=len(w)
    #w:边权,la:左部点顶标,lb:右部点顶标,va,vb:访问标记:是否在交错树中,mt:右部点匹配了哪一个左部点
    la,lb,mt,upd,va,vb,=[0]+[max(w[i][j] for j in range(1,n)) for i in range(1,n)],[0]*n,[0]*n,[inf]*n,[False]*n,[False]*n
    def dfs(x):
        va[x]=True
        for y in range(1,n):
            if not vb[y]:
                if la[x]+lb[y]-w[x][y]==0:
                    vb[y]=True
                    if mt[y]==0 or dfs(mt[y]):
                        mt[y]=x
                        return True
                else:upd[y]=min(upd[y],la[x]+lb[y]-w[x][y])
        return False
    for i in range(1,n):
        while True:
            for j in range(1,n):
                va[j],vb[j],upd[j]=False,False,inf
            if dfs(i):break
            delta=inf      
            for j in range(1,n):
                if not vb[j]:delta=min(delta,upd[j])
            for j in range(1,n):
                if va[j]:la[j]-=delta
                if vb[j]:lb[j]+=delta
    return -sum(w[mt[i]][i] for i in range(1,n))
class Solution:
    def minimumXORSum(self, nums1: List[int], nums2: List[int]) -> int:
        n=len(nums1)+1
        #预处理边权
        w=[[0]*n for _ in range(n)]
        for i,x in enumerate(nums1):
            for j,y in enumerate(nums2):
                w[i+1][j+1]=-(x^y)
        return KM(w)
```



```python
#KM算法(bfs),最小带权二分图匹配
def KM(w):
    """
    vb:记录右侧顶点是否已在交错树中
    slack:记录每个右侧顶点与当前交错树的差距
    last:记录交错树中右侧顶点的前驱，用于回溯增广路径
    y:从哑结点开始构造交错树
    mt[0]=i:假设左侧顶点 i 与哑结点 0 匹配
    x:当前 y 对应的左侧顶点
    """
    n=len(w)
    mt,la,lb=[0]*n,[0]+[max(w[i][j] for j in range(1, n)) for i in range(1, n)],[0] * n
    # 对于每个真实左侧顶点寻找匹配
    for i in range(1,n):
        vb,slack,last,y,mt[0]=[False]*n,[inf]*n,[0]*n,0,i
        while True:
            vb[y],x,delta,nexty=True,mt[y],inf,0
            # 更新所有未加入交错树右侧顶点的 slack 值
            for j in range(1,n):
                if not vb[j]:
                    if (d:=la[x]+lb[j]-w[x][j])<slack[j]:slack[j],last[j]=d,y
                    if slack[j]<delta:delta,nexty=slack[j],j
            # 如果 delta > 0，则调整所有标号
            if delta>0:
                for j in range(n):
                    if vb[j]:
                        la[mt[j]]-=delta
                        lb[j]+=delta
                    else:slack[j]-=delta
            y=nexty
            # 找到未匹配的右侧顶点时退出循环
            if mt[y]==0:break
        # 倒推更新增广路
        while y>0:mt[y],y=mt[last[y]],last[y]
    return -sum(w[mt[i]][i] for i in range(1,n))
class Solution:
    def minimumXORSum(self, nums1: List[int], nums2: List[int]) -> int:
        n=len(nums1)+1
        #预处理边权
        w=[[0]*n for _ in range(n)]
        for i,x in enumerate(nums1):
            for j,y in enumerate(nums2):
                w[i+1][j+1]=-(x^y)
        return KM(w)
```



# ACWing基础课:逆元

如果存在x使得$\frac{a}{b}\equiv a\cdot x(mod\ m)$成立,则x叫做b模m的逆元,记做$b^{-1}$

求逆元即对于b,寻找x使得$b\cdot x\equiv 1(mod\ m)$成立

费马定理:$b^{p-1}\equiv 1(mod\ p)$,所以$b\cdot b^{p-2}\equiv 1(mod\ p)$,所以$b^{p-2}$即为b的逆元,p为质数,p>=2,可以用快速幂求出

```c++
#include<iostream>
#include<algorithm>

using namespace std;

typedef long long LL;

//a^k%p
int qmi(int a,int k,int p)
{
    int res=1;
    while (k)
    {
        if (k&1) res=(LL)res*a%p;
        k>>=1;
        a=(LL)a*a%p;
    }
    return res;
}
int main()
{
    int n;
    scanf("%d",&n);
    while (n--)
    {
        int a,k,p;
        scanf("%d%d",&a,&p);
        
        int res=qmi(a,p-2,p);
        if (a%p) printf("%d\n",res);
        else puts("impossible");
        
    }
    return 0;
}
```



# ACWing基础课:组合数

```c++
//求组合数,递推
#include<iostream>
#include<algorithm>
using namespace std;
const int N=2010,mod=1e9+7;
int c[N][N];
void init()
{
    for (int i=0;i<N;i++)
        for (int j=0;j<=i;j++)
        	if (!j) c[i][j]=1;
    		else c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
    
}
int main()
{
    init();
    scanf("%d",&n);
    while (n--)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        printf("%d\n",c[a][b]);  
    }
    return 0;
}
```



```c++
//求组合数,预处理阶乘和逆元
#include<iostream>
#include<algorithm>

using namespace std;

typedef long long LL;
const int N=100010,mod=1e9+7;

//阶乘,逆元
int fact[N],infact[N];

int qmi(int a,int k,int p)
{
    int res=1;
    while (k)
    {
        if (k&1) res=(LL)res*a%p;
        a=(LL)a*a%p;
        k>>=1;
    }
    return res;
}

int main()
{
    fact[0]=infact[0]=1;
    for (int i=1;i<N;i++)
    {
        fact[i]=(LL)fact[i-1]*i%mod;
        infact[i]=(LL)infact[i-1]*qmi(i,mod-2,mod)%mod;
    }
    
    int n;
    scanf("%d",&n);
    while (n--)
    {
        int a,b;
        scanf("%d",&a,&b);
        printf("%d\n",(LL)fact[a]*infact[b]%mod*infact[a-b]%mod)
    }
    return 0;
}
```



卢卡斯定理:$C_a^b\equiv C_{a\ mod\ p}^{b\ mod \ p}\cdot C_{a/p}^{b/p}\ (mod \ p)$

```c++
#include<iostream>
#include<algorithm>
using namespace std;

typedef long long LL;

int p;

int qmi(int a,int k)
{
    int res=1;
    while (k)
    {
        if (k&1) res=(LL)res*a%p;
        a=(LL)a*a%p;
        k>>=1;
    }
    return res;
}

int C(int a,int b)
{
    int res=1;
    for (int i=1,j=a;i<=b;i++,j--)
    {
        res=(LL)res*j%p;
        res=(LL)res*qmi(i,p-2)%p;
    }
    return res;
}
int lucas(LL a,LL b)
{
    if (a<p && b<p) return C(a,b);
    return (LL)C(a%p,b%p)*lucas(a/p,b/p)%p;
}
int main()
{
    int n;
    cin>>n;
    while (n--)
    {
        LL a,b;
        cin>>a>>b>>p;
        cout<<lucas(a,b)<<endl;
    }
    return 0;
}
```



#扩展卢卡斯定理模版,用于解决C(n,k)%p,其中p不一定为质数

```c++
#include <cstdio>
using namespace std;
typedef long long LL;

void exgcd(int a,int b,LL &x,LL &y){
    if(b==0){
        x=1,y=0;
        return;
    }
    exgcd(b,a%b,y,x);
    y-=a/b*x;
}
LL inv(LL a,LL p){
    LL x,y;
    exgcd(a,p,x,y);
    return (x%p+p)%p;
}
LL qmi(LL a,LL b,LL p){
    LL res=1%p;
    while(b){
        if(b&1) res=res*a%p;
        b>>=1;
        a=a*a%p;
    }
    return res;
}
LL F(LL n,LL p,LL pk){
    if(n==0) return 1;
    LL rou=1,rem=1;//循环节，余项
    for(LL i=1;i<=pk;i++){
        if(i%p) rou=rou*i%pk;
    }
    rou=qmi(rou,n/pk,pk);
    for(LL i=pk*(n/pk);i<=n;i++){
        if(i%p) rem=rem*(i%pk)%pk;
    }
    return F(n/p,p,pk)*rou%pk*rem%pk;
}
LL G(LL n,LL p){
    if(n<p) return 0;
    return G(n/p,p)+(n/p);
}
LL cpk(LL n,LL m,LL p,LL pk){
    LL fz=F(n,p,pk);//分子
    LL fm1=inv(F(m,p,pk),pk);//分母1
    LL fm2=inv(F(n-m,p,pk),pk);//分母2
    LL mi=qmi(p,G(n,p)-G(m,p)-G(n-m,p),pk);//q^{x-y-z}
    return fz*fm1%pk*fm2%pk*mi%pk;//返回C(n,m)%p^k
}
LL A[1001],B[1001];//x=B(mod A)
LL exlucas(LL n,LL m,LL p){
    LL ljc=p,tot=0;
    for(LL i=2;i<=p/i;i++){
        if(ljc%i==0){
            LL pk=1;
            while(ljc%i==0){
                pk*=i;
                ljc/=i;
            }
            A[++tot]=pk;
            B[tot]=cpk(n,m,i,pk);
        }
    }
    if(ljc!=1){
        A[++tot]=ljc;
        B[tot]=cpk(n,m,ljc,ljc);
    }
    LL ans=0;
    for(LL i=1;i<=tot;i++){
        LL r=p/A[i],t=inv(r,A[i]);
        ans=(ans+B[i]*r%p*t%p)%p;
    }
    return ans;
}
int main(){
    LL n,m,p;
    scanf("%lld%lld%lld",&n,&m,&p);
    printf("%lld\n",exlucas(n,m,p));
    return 0;
}
```



```python
def exgcd(a, b):
    """
    扩展欧几里得算法，返回 (x, y)，使得 a*x + b*y = gcd(a, b)
    """
    if b == 0:
        return 1, 0
    else:
        x, y = exgcd(b, a % b)
        return y, x - (a // b) * y

def inv(a, p):
    """
    计算 a 在模 p 意义下的逆元
    """
    x, _ = exgcd(a, p)
    return (x % p + p) % p

def qmi(a, b, p):
    """
    快速幂，计算 a^b mod p
    """
    res = 1 % p
    while b:
        if b & 1:
            res = res * a % p
        b //= 2
        a = a * a % p
    return res

def F(n, p, pk):
    """
    递归计算 n! 去除 p 的因子后的值，结果 mod pk
    """
    if n == 0:
        return 1
    rou = 1
    rem = 1
    # 计算1到pk中不被p整除的数的乘积 mod pk
    for i in range(1, pk + 1):
        if i % p != 0:
            rou = rou * i % pk
    # rou 的指数部分
    rou = qmi(rou, n // pk, pk)
    # 计算剩余部分：i 从 pk*(n//pk) 到 n
    for i in range(pk * (n // pk), n + 1):
        if i % p != 0:
            rem = rem * (i % pk) % pk
    return F(n // p, p, pk) * rou % pk * rem % pk

def G(n, p):
    """
    计算 n! 中 p 的指数（即 p 的幂次）
    """
    if n < p:
        return 0
    return G(n // p, p) + n // p

def cpk(n, m, p, pk):
    """
    计算 C(n, m) mod pk，其中 pk 为 p 的幂次
    """
    fz = F(n, p, pk)
    fm1 = inv(F(m, p, pk), pk)
    fm2 = inv(F(n - m, p, pk), pk)
    mi = qmi(p, G(n, p) - G(m, p) - G(n - m, p), pk)
    return fz * fm1 % pk * fm2 % pk * mi % pk

def exlucas(n, m, p):
    """
    扩展Lucas定理：
    给定组合数 C(n, m) 和模数 p（可以是合数），通过分解 p 的质因子，
    分别计算每个质因子对应的结果，再用中国剩余定理合并结果。
    """
    ljc = p
    A = []
    B = []
    i = 2
    while i * i <= ljc:
        if ljc % i == 0:
            pk = 1
            while ljc % i == 0:
                pk *= i
                ljc //= i
            A.append(pk)
            B.append(cpk(n, m, i, pk))
        i += 1
    if ljc != 1:
        A.append(ljc)
        B.append(cpk(n, m, ljc, ljc))
    
    ans = 0
    # 中国剩余定理合并各个模数的答案
    for a, b in zip(A, B):
        r = p // a
        t = inv(r, a)
        ans = (ans + b * r % p * t % p) % p
    return ans

if __name__ == '__main__':
    import sys
    # 从标准输入读取 n, m, p
    data = sys.stdin.read().split()
    if len(data) >= 3:
        n = int(data[0])
        m = int(data[1])
        p = int(data[2])
        print(exlucas(n, m, p))
```



# AcWing基础课:卡特兰数

$C_{2n}^{n}-C_{2n}^{n-1}=\frac{C_{2n}^n}{n+1}$

```c++
#include <iostream>
#include <algorithm>

typedef long long LL;
using namespace std;

const int mod=1e9+7;

int qmi(int a,int k,int p)
{
    int res=1;
    while (k)
    {
        if (k&1) res=(LL)res*a%p;
        a=(LL)a*a%p;
        k>>=1;
    }
    return res;
}
int main()
{
    int n;
    cin>>n;
    
    int a=2*n,b=n;
    int res=1;
    
    for (int i=a;i>a-b;i--)res=(LL)res*i%mod;
    for (int i=1;i<=b;i++)res=(LL)res*qmi(i,mod-2,mod)%mod;
    res=(LL)res*qmi(n+1,mod-2,mod)%mod;
    cout<<res<<endl;
    return 0;
}
```



# AcWing基础课:博弈论

```c++
//先手必胜状态:可以走到某一个必败状态
//先手必败状态:走不到任何一个必败状态
//a1^a2^...^an=0先手必败
//a1^a2^...^an!=0先手必胜
```



```c++
#include<iostream>
#include<algorithm>

using namespace std;
int main()
{
    int n;
    int res=0;
    scanf("%d",&n);
    while (n--)
    {
        int x;
        scanf("%d",&x);
        res^=x;
    }
    if (res) puts("Yes");
    else puts("No");
    return 0;
}
```



```txt
mex运算:找到集合当中不存在的最小的自然数
SG(终点)=0
```



```c++
#include <iostream>
#include <cstring>
#include <algorithm>
#include <unordered_set>

using namespace std;
const int N=110,M=10010;

int n,m;
int s[N],f[M];

int sg(int x)
{
    if (f[x]!=-1) return f[x];
    unordered_set<int> S;
    for (int i=0;i<m;i++)
    {
        int sum=s[i];
        if (x>=sum) S.insert(sg(x-sum));
    }
    for (int i=0;;i++)
        if (!S.count(i))
            return f[x]=i;
}
int main()
{
    cin>>m;
    for (int i=0;i<m;i++) cin>>s[i];
    cin>>n;
    memset(f,-1,sizeof f);
    int res=0;
    for (int i=0;i<n;i++)
    {
        int x;
        cin>>x;
        res^=sg(x);
    }
    if (res) puts("Yes");
    else puts("No");
    return 0;
}
```

# AcWing基础课:计数dp

```c++
//整数划分
//1.转换成完全背包
#include <iostream>
#include <algorithm>

using namespace std;

const int N=1010,mod=1e9+7;

int n;
int f[N];

int main()
{
    cin>>n;
    f[0]=1;
	for (int i=1;i<=n;i++)
        for (int j=i;j<=n;j++)
            f[j]=(f[j]+f[j-i])%mod;
    cout<<f[n]<<endl;
    return 0;
}
```



```c++
//计数dp解法
//集合:所有总和是i,并且恰好表示成j个数的和的方案
//可以划分成两个集合:1.最小值是1,2.最小值大于1
//f[i][j]=f[i-1][j-1]+f[i-j][j]
//属性:数量

#include <iostream>
#include <algorithm>

using namespace std;

const int N=1010,mod=1e9+7;

int n;
int f[N][N];

int main()
{
    cin>>n;
    f[0][0]=1;
    for (int i=1;i<=n;i++)
        for (int j=1;j<=i;j++)
            f[i][j]=(f[i-1][j-1]+f[i-1][j])%mod;
    int res=0;
    for (int i=1;i<=n;i++) res=(res+f[n][i])%mod;
    
    cout<<res<<endl;
    
    return 0;
}
```

# AcWing基础课:状压dp

```c++
//蒙德里安的梦想
#include<cstring>
#include<iostream>
#include<algorithm>

using namespace std;

const int N=12,M=1<<N;

int n,m;
//f[i][j]表示摆放到第i列,上一列伸出来的小方格的状态是j的情况下,总共的方案数
long long f[N][M];
bool st[M];

int main()
{
    int n,m;
    while (cin>>n>>m,n||m)
    {
        memset(f,0,sizeof f);
        for (int i=0;i<1<<n;i++)
        {
            st[i]=true; //i状态不存在连续奇数个0
            int cnt=0;//cnt存储当前连续0的个数
            for (int j=0;j<n;j++)
                if (i>>j&1)
                {
                    if (cnt&1) st[i]=false;
                    cnt=0;
                }
            	else cnt++;
            if (cnt&1) st[i]=false;
        }
        f[0][0]=1;
        //枚举所有列
        for (int i=1;i<=m;i++)
            //枚举所有状态
            for (int j=0;j<1<<n;j++)
                //枚举i-1列所有状态
                for (int k=0;k<1<<n;k++)
                    if ((j&k)==0 && st[j|k])
                        f[i][j]+=f[i-1][k];
        cout<<f[m][0]<<endl;
    }
    return 0;
}
```



```c++
//最短Hamilton路径
//集合:所有从0走到j,走过的所有点是i的所有路径,i用二进制位表示
//状态计算,枚举倒数第二个点来分类
#include<cstring>
#include<iostream>
#include<algorithm>

using namespace std;

const int N=20,M=1<<N;

int n;
int w[N][N]; //两点之间距离
int f[M][N]; //状态

int main()
{
    cin>>n;
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
            cin>>w[i][j];
    memset(f,0x3f,sizeof f);
    f[1][0]=0;
    for (int i=0;i<1<<n;i++)
        for (int j=0;j<n;j++)
            if (i>>j&1)
                for (int k=0;k<n;k++)
                	if ((i-(1<<j))>>k&1)
                        f[i][j]=min(f[i][j],f[i-(1<<j)][k]+w[k][j])
    cout<<f[(1<<n)-1][n-1]<<endl;
    return 0;
}
```

# AcWing基础课:数位dp

```c++

```






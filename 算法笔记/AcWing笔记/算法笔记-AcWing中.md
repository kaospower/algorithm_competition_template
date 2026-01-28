# AcWing提高课:树形dp

```c++
// 1.求树的直径
#include <cstring>
#include <iostream>
#inlcude <algorithm>

using namespace std;
const int N=10010,M=N*2;

int n;
int h[N],e[M],w[M],ne[M],idx;
int ans;

void add(int a,int b,int c)
{
    e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
}

//dfs需要加上father参数
int dfs(int u,int father)
{
    int dist=0;//从当前点往下走的最大长度
    int d1=0,d2=0;
    for (int i=h[u];i!=-1;i=ne[i])
    {
        int j=e[i];
        if (j==father) continue;
        int d=dfs(j,u)+w[i];
        dist=max(dist,d);
        
        if (d>=d1) d2=d1,d1=d;
        else if (d>d2) d2=d;
    }
    ans=max(ans,d1+d2);
    return dist;
}
int main()
{
    cin>>n;
    memset(h,-1,sizeof h);
    for (int i=0;i<n-1;i++)
    {
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c),add(b,a,c);
    }
    dfs(1,-1);
    cout<<ans<<endl;
}

```



```c++
//2.求树的中心(换根dp/二次扫描)
//两遍dfs,第一遍用子节点更新父节点信息,第二遍用父节点更新子节点信息

#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;
const int N=10010,M=N*2,INF=0x3f3f3f3f;

int n;
int h[N],e[M],w[M],ne[M],idx;
int d1[N],d2[N],p1[N],p2[N],up[N];

//加边函数
void add(int a,int b,int c)
{
    e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
}

int dfs_d(int u,int father)
{
    d1[u]=d2[u]=-INF;
    for (int i=h[u];i!=-1;i=ne[i])
    {
        int j=e[i];
        if (j==father) continue;
        int d=dfs_d(j,u)+w[i];
        if (d>=d1[u]){
        	d2[u]=d1[u],d1[u]=d;
            p2[u]=p1[u],p1[u]=j;
        } 
        else if (d>d2[u]) d2[u]=d,p2[u]=j;
    }
    if (d1[u]=-INF) d1[u]=d2[u]=0;
    return d1[u];
}

void dfs_u(int u,int father)
{
    for (int i=h[u];i!=-1;i=ne[i])
    {
        int j=e[i];
        if (j==father) continue;
        
        if (p1[u]==j) up[j]=max(up[u],d2[u])+w[i];
        else up[j]=max(up[u],d1[u])+w[i];
        
        dfs_u(j,u)
    }
}
int main()
{
    cin>>n;
    memset(h,-1,sizeof h);
    for (int i=0;i<n-1;i++)
    {
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c),add(b,a,c);
    }
    dfs_d(1,-1);
    dfs_u(1,-1);
    int res=INF;
    //d1[i]代表每个点往下走的最远距离,up[i]代表每个点往上走的最远距离
    for (int i=1;i<=n;i++) res=min(res,max(d1[i],up[i]));
    printf("%d\n",res);
    return 0;
}
```

# AcWing提高课:最近公共祖先

```txt
LCA:最近公共祖先
一个点本身也是祖先
1.向上标记法 O(n)
2.倍增法 O(mlogn) m是查询次数,n是节点数量
3.tarjan-离线求LCA O(n+m)
预处理f[i,j]从i开始向上走2^j步所能走到的节点.0<=j<=logn
j=0:f(i,j)=i的父节点
j>0:f(i,j)=f(f(i,j-1),j-1)
depth[i]表示深度
哨兵:如果从i开始跳2^j步会跳过根节点,那么f[i,j]=0;depth[0]=0
步骤:
1.先将两个点跳到同一层,基于二进制拼凑的思想,差的步数:depth[x]-depth[y],即枚举depth[f(x,k)]>=depth[y]
2.让两个点同时往上跳,一直跳到它们的最近公共组先的下面一层,即判断f(a,k)!=f(b,k),注意不能用f(a,k)==f(b,k)判断,因为这样得到的不一定是最近公共组先,有可能是更远的公共组先,最后结果即为f(x,0),即从当前位置再向上跳2^0(1)步就是最近公共组先
预处理O(nlogn)
查询时间复杂度O(logn)
```



```c++
//祖孙询问
//倍增求LCA模版
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=40010,M=N*2;

int n,m;
int h[N],e[M],ne[M],idx;
int depth[N],fa[N][16];
int q[N];

void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}

void bfs(int root)
{
    memset(depth,0x3f,sizeof depth);
    depth[0]=0,depth[root]=1;
    int hh=0,tt=0;
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i]){
            int j=e[i];
            if (depth[j]>depth[t]+1)
            {
                depth[j]=depth[t]+1;
                q[++tt]=j;
                fa[j][0]=t;
                for (int k=1;i<=15;k++)
                	fa[j][k]=fa[fa[j][k-1]][k-1];
            }
        }
    }
}

int lca(int a,int b)
{
    //先跳到同一层
    if (depth[a]<depth[b]) swap(a,b);
    for (int k=15;k>=0;k--)
        if (depth[fa[a][k]]>=depth[b])
            a=fa[a][k];//如果跳出根节点的话,由于我们设置了哨兵depth[0]=0,因此条件会不成立
    if (a==b) return a;
    for (int k=15;k>=0;k--)
        if (fa[a][k]!=fa[b][k])
        {
            a=fa[a][k];
            b=fa[b][k];
        }
    return fa[a][0];
}
int main()
{
    scanf("%d",&n);
    int root=0;
    memset(h,-1,sizeof h);
    
    for (int i=0;i<n;i++)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        if (b==-1) root=a;
        else add(a,b),add(b,a);
    }
    
    bfs(root);
    scanf("%d",&m);
    while (m--)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        int p=lca(a,b);
        if (p==a) puts("1");
        else if (p==b) puts("2");
        else puts("0");
    }
    return 0;
}
```



```txt
在线做法:每读入一个询问就给出一个输出
离线做法:一次读入所有询问,然后统一给出所有输出

Tarjan-离线求LCA(本质是对向上标记法的优化)
时间复杂度O(n+m),n是节点数量,m是查询次数

向上标记法:从一个点a向根节点走,将途经的所有点进行标记,之后从另一个点b向根节点走,如果遇到之前标记的点c,那么这个点c就是a和b的最近公共组先

tarjan:
dfs时将所有节点分成三大类:
1.已经遍历过且回溯过的点:2
2.正在搜索的分支:1
3.还未搜索到的点:0

对于某个正在搜索的点a,我们处理所有和a相关的询问,a和某个处于已经遍历过且回溯过的分支的LCA,即为该分支的祖先,
即其在并查集中的代表元

树上任意两点x,y之间的距离:dis(x,y)=d(x)+d(y)-2d(p),p为x,y的LCA,d(x)为x到根节点之间的距离
```



```c++
//距离
//tarjan算法求LCA模版
#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace std;

typedef pair<int,int>PII;

const int N=20010,M=N*2;

int n,m;
int h[N],e[M],w[M],ne[M],idx;
int dist[N];//每个点和根节点之间的距离
int p[N];
int res[N];//所有询问结果
int st[N];//标记数组
vector<PII>query[N]; //first存查询的另外一个点,second存查询编号

void add(int a,int b,int c)
{
    e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
}

void dfs(int u,int fa) //fa代表当前点的父节点
{
    for (int i=h[u];~i;i=ne[i])
    {
        int j=e[i];
        if (j==fa) continue;
        dist[j]=dist[u]+w[u];
        dfs(j,u)
    }
}

int find(int x)
{
    if (p[x]!=x) p[x]=find(p[x]);
    return p[x];
}
void tarjan(int u)
{
    st[u]=1;//正在搜索的分支
    for (i=h[u];~i;i=ne[i])
    {
        int j=e[i];
        //还未遍历到的点
        if (!st[j]){
            tarjan(j);
            //并查集合并
            p[j]=u;
        }
    }
    for (auto item:query[u]){
        int y=item.first,id=item.second;
        if (st[y]==2)
        {
            //最近公共祖先
            int anc=find(y);
            res[id]=dist[y]+dist[u]-2*dist[anc];
        }
    }
    st[u]=2;
}

int main(){
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof h);
    for (int i=0;i<n-1;i++){
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c),add(b,a,c);
    }
    for (int i=0;i<m;i++)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        //如果a==b,结果就是res数组默认值0
        if (a!=b){
        	query[a].push_back({b,i});
            query[b].push_back({a,i});
        }
        
    }
    for (int i=1;i<=n;i++) p[i]=i; //初始化并查集
    dfs(1,-1);
    tarjan(1);
    for (int i=0;i<m;i++) printf("%d\n",res[i]);
    return 0;
}
```



```txt
求LCA的另一种做法:基于RMQ的做法
将整棵树dfs一遍,遍历时记录dfs序
例如下面这棵二叉树:
			1
          /   \
         2     3
        / \   /  \
       4   5 6    7
      / \
     8   9
其dfs序为12484942521363731
如果我们求8和5的LCA,取出8到5这段:849425,其中最小值2即为8和5的LCA,转化成了区间最小值问题,可以用RMQ算法求解(倍增,线段树等)
这种解法更麻烦,一般不使用
```



```txt
定理:对于一张无向图,如果存在最小生成树和(严格)次小生成树,那么对于任何一棵最小生成树,都存在一棵(严格)次小生成树,使得这两棵树只有一条边不同.
非严格次小生成树:枚举每一条非树边w,然后加上这条边w,再减去最大权值的边wi,即sum+w-wi
严格次小生成树:我们如果发现最大边权和w相等,还需要枚举次大边权,如果次大边权<w,我们就可以替换次大边权
```



```txt
预处理
fa(i,j):从i往上跳2^j步可以到达的点
d1(i,j):从i往上跳2^j步路径上的最大边权
d2(i,j):从i往上跳2^j步路径上的次大边权

每一段都记录最大值和次大值
```



```c++
//秘密的牛奶运输
//次小生成树模版
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>

using namespace std;

typedef long long LL;
const int N=100010,M=300010,INF=0x3f3f3f3f;

int n,m;
struct Edge
{
    int a,b,w;
    bool used;
    bool operator<(const Edge &t)> const
    {
        return w<t.w;
    }
}edge[M];
int p[N];//并查集数组
int h[N],e[M],w[M],ne[M],idx;
int depth[N],fa[N][17],d1[N][17],d2[N][17];
int q[N];//bfs队列

void add(int a,int b,int c)
{
    e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
}
int find(int x){
    if (x!=p[x]) p[x]=find(p[x]);
    return p[x];
}
LL kruskal()
{
    //预处理并查集
    for (int i=1;i<=n;i++) p[i]=i;
    sort(edge,edge+m);
    LL res=0;
    for (int i=0;i<m;i++){
        int a=find(edge[i].a),b=find(edge[i].b),w=edge[i].w;
        //并查集合并
        if (a!=b)
        {
            p[a]=b;
            res+=w;
            edge[i].used=true;
        }
    }
    return res;
}

void build()
{
    memset(h,-1,sizeof h);
    for (int i=0;i<m;i++){
        if (edge[i].used){
            int a=edge[i].a,b=edge[i].b,w=edge[i].w;
            add(a,b,w),add(b,a,w);
        }
    }
}
//预处理倍增数组
void bfs()
{
    memset(depth,0x3f,sizeof depth);
    depth[0]=0,depth[1]=1; //0是哨兵,1是根节点
    q[0]=1;
    int hh=0,tt=0;
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i]){
            int j=e[i];
            if (depth[j]>depth[t]+1){
                //depth默认值是inf,如果depth[j]>depth[t]+1,说明还没访问过j
                depth[j]=depth[t]+1;
                q[++tt]=j;
                fa[j][0]=t;//从j向上走2^0(1)步是t
                d1[j][0]=w[i],d2[j][0]=-INF;//最大边权和次大边权
                for (int k=1;k<=16;k++){
                    int anc=fa[j][k-1]
                    fa[j][k]=fa[anc][k-1];
                    int distance[4]={d1[j][k-1],d2[j][k-1],d1[anc][k-1],d2[anc][k-1]};
                    d1[j][k]=d2[j][k]=-INF;
                    for (int u=0;u<4;u++){
                        int d=distance[u];
                        if (d>d1[j][k]) d2[j][k]=d1[j][k],d1[j][k]=d;
                        else if (d!=d1[j][k] && d>d2[j][k]) d2[j][k]=d;
                    }
                }
            }
        }
    }
}

int lca(int a,int b,int w)
{
    static int distance[N*2];
    int cnt=0;
    if (depth[a]<depth[b]) swap(a,b);
    for (int k=16;k>=0;k--)
        if (depth[fa[a][k]]>=depth[b])
        {
            distance[cnt++]=d1[a][k];
            distance[cnt++]=d2[a][k];//存储最大值和次大值
            a=fa[a][k];
        }
    if (a!=b)
    {
        for (int k=16;k>=0;k--)
            if (fa[a][k]!=fa[b][k])
            {
                distance[cnt++]=d1[a][k];
                distance[cnt++]=d2[a][k];
                distance[cnt++]=d1[b][k];
                distance[cnt++]=d2[b][k];
                a=fa[a][k],b=fa[b][k];
            }
        distance[cnt++]=d1[a][0];
        distance[cnt++]=d1[b][0];
    }
    int dist1=-INF,dist2=-INF;
    for (int i=0;i<cnt;i++)
    {
        int d=distance[i];
        if (d>dist1) dist2=dist1,dist1=d;
        else if (d!=dist1 && d>dist2) dist2=d;
        
    }
    if (w>dist1) return w-dist1;
    if (w>dist2) return w-dist2;
    return INF;
}
int main()
{
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof h);
    for (int i=0;i<m;i++)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        edge[i]={a,b,c};
    }
    LL sum=kruskal();
    build();
    bfs();
    
    LL res 1e18;//long long范围正无穷
    for (int i=0;i<m;i++)
    {
        if (!edge[i].used)
        {
            int a=edge[i].a,b=edge[i].b,w=edge[i].w;
            res=min(res,sum+lca(a,b,w));
        }
    }
    printf("%lld\n",res);
    return 0;
}
```



```txt
树上差分
如果想对树上x和y之间的所有边都加上c
d(x)+=c,d(y)+=c,d(p)-=2c,p是x,y的LCA
```



```c++
//暗之连锁
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=100010,M=N*2;

int n,m;
int h[N],e[M],ne[M],idx;
int depth[N],fa[N][17];
int d[N];//存储差分值
int q[N];
int ans;

void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}

//预处理倍增数组
void bfs()
{
    memset(depth,0x3f,sizeof depth);
    depth[0]=0,depth[1]=1;
    int hh=0,tt=0;
    q[0]=1;
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int j=e[i];
            if (depth[j]>depth[t]+1)
            {
                depth[j]=depth[t]+1;
                q[++tt]=j;
                fa[j][0]=t;
                for (int k=1;i<=16;k++)
                {
                    fa[j][k]=fa[fa[j][k-1]][k-1];
                }
            }
        }
    }
}
int lca(int a,int b)
{
    if (depth[a]<depth[b]) swap(a,b);
    for (int k=16;k>=0;k--)
        if (depth[fa[a][k]]>=depth[b])
            a=fa[a][k];
    if (a==b) return a;
    for (int k=16;k>=0;k--)
        if (fa[a][k]!=fa[b][k])
        {
            a=fa[a][k];
            b=fa[b][k];
        }
    return fa[a][0];
}
int dfs(int u,int father)
{
    int res=d[u];
    for (int i=h[u];~i;i=ne[i])
    {
        int j=e[i];
        if (j!=father)
        {
            int s=dfs(j,u);
            if (s==0) ans+=m;
            else if (s==1) ans++;
            res+=s;
        }
    }
    return res;
}
int main()
{
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof h);
    for (int i=0;i<n-1;i++)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        add(a,b),add(b,a);
    }
    bfs();
    for (int i=0;i<m;i++)
    {
        int a,b;
        scanf("%d%d",&a,&b);//读入非树边
        int p=lca(a,b);
        d[a]++,d[b]++,d[p]-=2;//树上差分
    }
    dfs(1,-1)
    printf("%d\n",ans);
    return 0;
}
```



# AcWing提高课:有向图的强连通分量

**连通分量**:对于一个有向图,对于分量中任意两点u,v,必然可以从u走到v,且从v走到u。

**强连通分量(SCC)**:极大连通分量,即加上任何其他点都不再是连通分量了

强连通分量可以通过缩点方式将有向图转化成有向无环图(DAG)/拓扑图

**缩点**:将所有强连通分量缩成一个点



将所有边分成四大类:

1.树枝边(特殊的前向边),x是y的父节点

2.前向边(x,y),x是y的祖先节点

3.后向边(x,y),y是x的祖先节点

4.横叉边(x,y),连向其他分支的边(左边的分支)



某个点是否在强连通分量中

1.存在后向边指向祖先节点

2.先走到横叉边,横叉边再走到祖先节点



**tarjan算法求强连通分量**

时间戳:按照dfs遍历的顺序给每个点一个编号

对每个点定义两个时间戳:

dfn[u]表示遍历到u的时间戳

low[u]表示从u开始走,所能遍历到的最小时间戳

u是其所在强连通分量的最高点,等价于dfn[u]==low[u]

时间复杂度O(n+m)

缩点:遍历所有点和其所有邻点j,如果i,j不在同一个scc中,加一条新边id(i)->id(j),可以将有向图转化成有向无环图,利用DAG的拓扑序,可以解决很多问题

连通分量编号递减的顺序一定是拓扑序

```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

void tarjan(int u)
{
    dfn[u]=low[u]=++timestamp;
    stk[++top]=u,in_stk[u]=true;
    for (int i=h[u];~i;i=ne[i])
    {
        int j=e[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u]=min(low[u],low[j]);
        }
        else if (in_stk[j])
            low[u]=min(low[u],dfn[j]);
    }
    if (dfn[u]==low[u])
    {
        int y;
        ++scc_cnt;
        do{
            y=stk[top--];
            in_stk[y]=false;
            id[y]=scc_cnt;
        }while (y!=u);
    }
}
```



```c++
//受欢迎的奶牛
//利用tarjan+缩点将原图转化成DAG,然后查看出度为0的点是否为1个
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>

using namespace std;

const int N=10010,M=50010;
int n,m;
int h[N],e[M],ne[M],idx;
int dfn[N],low[N],timestamp;
int stk[N],top;//栈
bool in_stk[N];//表示每个点是否在栈中
//id表示每个点属于哪个强连通分量编号,scc_cnt为强连通分量数量
//size表示每个强连通分量中点的数量
int id[N],scc_cnt,size[N];
int dout[N];//每个强连通分量的出度

void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}
void tarjan(int u)
{
    dfn[u]=low[u]=++timestamp;
    stk[++top]=u,in_stk[u]=true;
    for (int i=h[u];~i;i=ne[i])
    {
        int j=ne[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u]=min(low[u],low[j]);
        }
        else if (in_stk[j]) low[u]=min(low[u],dfn[j]);
    }
    if (dfn[u]==low[u])
    {
        ++scc_cnt;
        int y;
        do
        {
            y=stk[top--];
            in_stk[y]=false;
            id[y]=scc_cnt;
            size[scc_cnt]++;
        }while (y!=u);
    }
}
int main()
{
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        add(a,b);
    }
    for (int i=1;i<=n;i++)
        if (!dfn[i])
            tarjan(i);
    for (int i=1;i<=n;i++)
        for (int j=h[i];~j;j=ne[j])
        {
            int k=e[j];
            int a=id[i],b=id[k];
            if (a!=b) dout[a]++;
        }
    int zeros=0,sum=0;//sum表示所有出度为0的强连通分量的点的数量之和
    for (int i=1;i<=scc_cnt;i++)
        if (!dout[i])
        {
            zeros++;
            sum+=size[i];
            if (zeros>1)
            {
                sum=0;
                break;
            } 
        }
    printf("%d\n",sum);
    return 0;
}
```



```c++
//学校网络
//此题思路是将原图转化成DAG,然后答案为max(p,q),p为所有入度为0的点的数量,q为所有出度为0的点的数量
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=110,M=10010;

int n;
int h[N],e[M],ne[M],idx;
int dfn[N],low[N],timestamp;
int stk[N],top;
bool in_stk[N];
int id[N],scc_cnt;
int din[N],dout[N];//din统计入度,dout统计出度

void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}
void tarjan(int u)
{
    dfn[u]=low[u]=++timestamp;
    stk[++top]=u,in_stk[u]=true;
    for (int i=h[u];~i;i=ne[i])
    {
        int j=e[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u]=min(low[u],low[j]);
        }
        else if (in_stk[j]) low[u]=min(low[u],dfn[j]);
    }
    if (dfn[u]==low[u])
    {
        ++scc_cnt;
        int y;
        do
        {
            y=stk[top--];
            in_stk[y]=false;
            id[y]=scc_cnt;
        }while (y!=u);
    }
}
int main()
{
    cin>>n;
    memset(h,-1,sizeof h);
    for (int i=1;i<=n;i++)
    {
        int t;
        while (cin>>t,t) add(i,t);
    }
    for (int i=1;i<=n;i++) 
        if (!dfn[i])
        	tarjan(i);
    for (int i=1;i<=n;i++)
        for (int j=h[i];~j;j=ne[j])
        {
            int k=e[j];
            int a=id[i],b=id[k];
            if (a!=b)
            {
                dout[a]++;
                din[b]++;
            }
        }
    int a=0,b=0;
    for (int i=1;i<=scc_cnt;i++)
    {
        if (!dfn[i]) a++;
        if (!dout[i]) b++;
    }
    printf("%d\n",a);
    if (scc_cnt==1) puts("0");
    else printf("%d\n",max(a,b));
    return 0;
}
```



```c++
//最大半连通子图
//1.tarjan,2.缩点,建图,给边判重,3.按照拓扑序递推
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <unordered_set>

using namespace std;

const int N=100010,M=1000010;
typedef long long LL;

int n,m;
int h[N],hs[N],e[M],ne[M],idx;//h:原图表头,hs:缩点之后表头
int dfs[N],low[N],timestamp;
int stk[N],top;
bool in_stk[N];
int id[N],scc_cnt,size[N];
int f[N],g[N];

void add(int h[],int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}
void tarjan(int u)
{
    dfn[u]=low[u]=++timestamp;
    stk[++top]=u,in_stk[u]=true;
    
    for (int i=h[u];~i;i=ne[i])
    {
        int j=e[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u]=min(low[u],low[j]);
        }
        else if (in_stk[j]) low[u]=min(low[u],dfn[j]);
    }
    if (dfn[u]==low[u])
    {
        ++scc_cnt;
        int y;
        do{
            y=stk[top--];
            in_stk[y]=false;
            id[y]=scc_cnt;
            size[scc_cnt]++;
        }while (y!=u);
    }
}
int main()
{
    memset(h,-1,sizeof h);
    memset(hs,-1,sizeof hs);
    scanf("%d%d",&n,&m,&mod);
    while (m--)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        add(h,a,b);
    }
    for (int i=1;i<=n;i++)
        if (!dfn[i])
            tarjan(i);
    unordered_set<LL>S;//哈希函数,(u,v)->u*1000000+v
    for (int i=1;i<=n;i++)
        for (int j=h[j];~j;j=ne[j])
        {
            int k=e[j];
            int a=id[i],b=id[k];
            LL hash=a*1000000ll+b;
            if (a!=b && !S.count(hash))
            {
                add(hs,a,b);
                S.insert(hash);
            }
        }
    for (int i=scc_cnt;i;i--)
    {
        if (!f[i])
        {
            f[i]=size[i];
            g[i]=1;
        }
        for (int j=hs[i];~j;j=ne[j])
        {
            int k=e[j];
            if (f[k]<f[i]+size[k])
            {
                f[k]=f[i]+size[k];
                g[k]=g[i];
            }
            else if (f[k]==f[i]+size[k])
                g[k]=(g[k]+g[i])%mod;
        }
    }
    int maxf=0,sum=0;
    for (int i=1;i<=scc_cnt;i++)
        if (f[i]>maxf)
        {
            maxf=f[i];
            sum=g[i];
        }
    	else if (f[i]==maxf) sum=(sum+g[i])%mod;
    printf("%d\n",maxf);
    printf("%d\n",sum);
    return 0;
    
}
```



```c++
//银河
//1.tarjan,2.缩点+建图,3.依据拓扑序递推
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>

using namespace std;

typedef long long LL;

const int N=100010,M=600010;

int n,m;
int h[N],hs[N],e[M],ne[M],w[M],idx;
int dfs[N],low[N],timestamp;
int stk[N],top;
bool in_stk[N];
int id[N],scc_cnt,size[N];
int dist[N];

void add(int h[],int a,int b,int c)
{
    e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
}

void tarjan(int u)
{
    dfn[u]=low[u]=++timestamp;
    stk[++top]=u,in_stk[u]=true;
    for (int i=h[u];~i;i=ne[i])
    {
        int j=e[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u]=min(low[u],low[j]);
        }
        else if (in_stk[j]) low[u]=min(low[u],dfn[j]);
    }
    if (dfn[u]==low[u])
    {
        ++scc_cnt;
        int y;
        do{
            y=stk[top--];
            in_stk[y]=false;
            id[y]=scc_cnt;
            size[scc_cnt]++;
        }while (y!=u);
    }
}
int main()
{
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof h);
    memset(hs,-1,sizeof hs);
    
    for (int i=1;i<=n;i++) add(h,0,i,1);
    while (m--)
    {
        int t,a,b;
        scanf("%d%d%d",&t,&a,&b);
        if (t==1) add(h,b,a,0),add(h,a,b,0);
        else if (t==2) add(h,a,b,1);
        else if (t==3) add(h,b,a,0);
        else if (t==4) add(h,b,a,1);
        else add(h,a,b,0);
    }
    tarjan(0);
    bool success=true;
    for (int i=0;i<=n;i++)
    {
        for (int j=h[i];~j;j=ne[j])
        {
            int k=e[j];
            int a=id[i],b=id[k];
            if (a==b)
            {
                if (w[j]>0){
                    success=false;
                    break;
                }
            }
            else add(hs,a,b,w[i]);
        }
        if (!success) break;
    }
    if (!success) puts("-1");
    else
    {
        for (int i=scc_cnt;i;i--)
            for (int j=hs[i];~j;j=ne[j])
            {
                int k=e[j];
                dist[k]=max(dist[k],dist[i]+w[j]);
            }
        LL res=0;
        for (int i=1;i<=scc_cnt;i++) res+=(LL)dist[i]*size[i];
        printf("%lld\n",res);
    }
    return 0;
}
```

# AcWing提高课:无向图的双连通分量

**桥(割边)**:对于无向连通图的某一条边,如果把这条边删掉图变得不连通,就称为桥
**割点**:在连通的无向图当中,如果把某一个点删除之后整个图变得不连通,就称为割点

每个割点都至少属于两个连通分量

**极大的连通分量**:对于一个连通分量,如果不存在一个包含它且点数比它多的连通分量,这个连通分量就称为极大的连通分量
$$
双连通分量
\begin{equation*}
\begin{cases}
1.边双连通分量(e-dcc):极大的不包含桥的连通块\\
2.点双连通分量(v-dcc):极大的不包含割点的连通块
\end{cases}
\end{equation*}
$$

边的双连通分量问题
时间戳,dfn(x):dfs时第一次到达某个点的时间点

low(x):以x为根的子树,往下走,所能到达的最早的一个点

无向图不存在横叉边

$x\rightarrow y是桥\Leftrightarrow dfn(x)<low(y)$

寻找边的双连通分量:

方法1.将所有桥删掉

方法2.利用栈

```c++
//边双连通分量算法
//冗余路径
//给定一个无向连通图,问最少加几条边,可以将其变成一个边双连通分量
//将原图缩点成树,ans=(cnt+1)/2下取整,cnt为缩完点之后度数为1的点的个数
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=5010,M=20010;

int n,m;
int h[N],e[M],ne[M],idx;
int dfn[N],low[N],timestamp;
int stk[N],top;//栈
int id[N],dcc_cnt;//记录每个点属于哪个双连通分量
bool is_bridge[M];//记录每条边是不是桥
int d[N];//度数

void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}
void tarjan(int u,int from)
{
    dfn[u]=low[u]=++timestamp;
    stk[++top]=u;
    for (int i=h[u];~i;i=ne[i])
    {
        int j=e[i];
        if (!dfn[j])
        {
            tarjan(j,i);
            low[u]=min(low[u],low[j]);
            if (dfn[u]<low[j])
                is_bridge[i]=is_bridge[i^1]=true;    
        }
        else if (i!=(from^1))
            low[u]=min(low[u],dfn[j]);
    }
    if (dfn[u]==low[u])
    {
        ++dcc_cnt;
        int y;
        do{
            y=stk[top--];
            id[y]=dcc_cnt;
        }while (y!=u);
    }
}
int main()
{
    cin>>n>>m;
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b;
        cin>>a>>b;
        add(a,b),add(b,a);
    }
    tarjan(1,-1);//记录父节点
    for (int i=0;i<idx;i++)
        if (is_bridge[i])
            d[id[e[i]]]++;
    int cnt=0;
    for (int i=1;i<=dcc_cnt;i++)
        if (d[i]==1)
            cnt++;
    printf("%d\n",(cnt+1)/2);
    return 0;
}
```



点双连通分量算法

**如何求割点**:

对于从x到y的边

low(y)>=dfn(x)

1.如果x不是根节点,那么x是割点

2.如果x是根节点,至少有两个子节点$y_i$满足$low(y_i)>=dfn(x)$

**如何求点双连通分量**:

利用栈

```c++
//电力
//求删除无向图的一个点之后,连通块最多有多少
//1.统计连通块个数cnt,
//2.枚举从哪个块中删,删除哪个点,设删除该点后可以将该连通块分成s部分,答案即为max(cnt-1+s)
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=10010,M=30010;

int n,m;
int h[N],e[M],ne[M],idx;
int dfn[N],low[N],timestamp;
int root,ans;
void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}
void tarjan(int u)
{
    dfn[u]=low[u]=timestamp;
    int cnt=0;
    for (int i=h[u];~i;i=ne[i])
    {
        int j=e[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u]=min(low[u],low[j]);
            if (low[j]>=dfn[u]) cnt++;
        }
        else low[u]=min(low[u],dfn[j]);
    }
    if (u!=root && cnt) cnt++;
    ans=max(ans,cnt);
}
int main()
{
    while (scanf("%d%d",&n,&m),n||m)
    {
        memset(dfn,0,sizeof dfn);//dfn兼具时间戳和判重数组的作用
        memset(h,-1,sizeof h);
        idx=timestamp=0;
        while (m--)
        {
            int a,b;
            scanf("%d%d",&a,&b);
            add(a,b),add(b,a);
        }
        ans=0;
        //统计连通块数量
        int cnt=0;
        for (int root=0;root<n;root++)
            if (!dfn[root])
            {
                cnt++;
                tarjan(root);
            }
        printf("%d\n",ans+cnt-1);
    }
    return 0;
}
```



```c++
//矿场搭建
//点双连通分量算法
//给定一个无向图,问最少在几个点上设置出口,可以使得不管其他哪个点坍塌,其余所有点都可以与某个出口连通
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace std;

typedef unsigned long long ULL;
const int N=1010,M=510;

int n,m;
int h[N],e[M],ne[M],idx;
int dfn[N],low[N],idx;
int stk[N],top;
int dcc_cnt;
vector<int> dcc[N]; //双连通分量里面有哪些点
bool cut[N];//双连通分量是不是割点
int root;

void add(int a,int b)
{
    e[idx]=b,ne[idx]=h[a],h[a]=idx++;
}
void tarjan(int u)
{
    dfn[u]=low[u]=++timestamp;
    stk[++top]=u;
    if (u==root && h[u]==-1)
    {
        dcc_cnt++;
        dcc[dcc_cnt].push_back(u);
        return;
    }
    int cnt=0;
    for (int i=h[u];~i;i=ne[i])
    {
        int j=e[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u]=min(low[u],low[j]);
            if (dfn[u]<=low[j])
            {
                cnt++;
                if (u!=root || cnt>1) cut[u]=true;
               	++dcc_cnt;
                int y;
                do{
                    y=stk[top--];
                    dcc[dcc_cnt].push_back(y);
                }while (y!=j);
                dcc[dcc_cnt].push_back(u);
            }
        }
        else low[u]=min(low[u],low[j]);
    }
}
int main()
{
    int T=1;
    while (cin>>m,m)
    {
        for (int i=1;i<=dcc_cnt;i++) dcc[i].clear();
        idx=n=timestamp=top=dcc_cnt=0;
        memset(h,-1,sizeof h);
        memset(dfn,0,sizeof dfn);
        memset(cut,0,sizeof cut);
        while (m--)
        {
            int a,b;
            cin>>a>>b;
            n=max(n,a),n=max(n,b);
            add(a,b),add(b,a);
        }
        for (root=1;root<=n;root++)
            if (!dfn[root])
                tarjan(root);
        int res=0;
        ULL num=1;
        for (int i=1;i<=dcc_cnt;i++)
        {
            int cnt=0;
            for (int j=0;j<dcc[i].size();j++)
                if (cut[dcc[i][j]])
                    cnt++;
            if (cnt==0) res+=2,num*=dcc[i].size()*(dcc[i].size()-1)/2;
            else if (cnt==1) res++,num*=dcc[i].size()-1;
        }
        printf("Case %d: %d %llu\n",T++,res,num);   
    }
    return 0;
}
```



# AcWing提高课:二分图

```txt
注意:更复杂的二分图问题要用网络流来解决
1.二分图<=>不存在奇数环<=>染色法不存在矛盾
增广路径:从非匹配点出发,经过非匹配边,匹配边,非匹配边,匹配边...非匹配边,最后走到非匹配点
最大匹配等价于不存在增广路径
```



```txt
最小点覆盖:
在二分图中最小点覆盖=最大匹配数
```



# AcWing提高课:平衡树treap

```mysql
treap=tree+heap
常用平衡树:C++ set,map,treap,splay,sbt,AVL,红黑树等
treap,splay最常用
```

# AcWing提高课:AC自动机

AC自动机=Trie+KMP
还有一种优化成Trie图的写法

AC自动机相当于在Trie上建立next数组



AC自动机时间复杂度O(n),n为单词数量

```c++
//AC自动机原始模版
//搜索关键词
//给定n个长度不超过50的由小写英文字母组成的单词,以及一篇长为m的文章,请问,有多少个单词在文章中出现了
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>
using namespace std;

const int N=10010,S=55,M=1000010;
int n;
int tr[N*S][26],cnt=[N*S],idx;//cnt表示以每个节点结尾的单词数量
char str[M];
int q[N*S],ne[N*S];//q为bfs队列,ne为next数组

void insert()
{
    int p=0;
    for (int i=0;str[i];i++)
    {
        int t=str[i]-'a';
        if (!tr[p][t]) tr[p][t]=++idx;
        p=tr[p][t];
    }
    cnt[p]++;
}
void build()
{
    int hh=0,tt=-1;
    for (int i=0;i<26;i++)
        if (tr[0][i])
        	q[++tt]=tr[0][i];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=0;i<26;i++)
        {
            int c=tr[t][i];
            if (!c) continue;
            int j=ne[t];
            while (j && !tr[j][i]) j=ne[j];
            if (tr[j][i]) j=tr[j][i];
            ne[c]=j;
            q[++tt]=c;
        }
    }
}
int main(){
    int T;
    scanf("%d",&T);
    while (T--){
        memset(tr,0,sizeof tr);
        memset(cnt,0,sizeof cnt);
        memset(ne,0,sizeof ne);
        idx=0;
        
        scanf("%d",&n);
        for (int i=0;i<n;i++){
            scanf("%s",str);
            insert();
        }
        
        build();
        scanf("%s",str);
        int res=0;
        for (int i=0,j=0;str[i];i++)
        {
            int t=str[i]-'a';
            while (j && !tr[j][t]) j=ne[j];
            if (tr[j][t]) j=tr[j][t];
            
            int p=j;
            while (p)
            {
                res+=cnt[p];
                cnt[p]=0;
                p=ne[p];
            }
        }
        printf("%d\n",res);
    }
    return 0;
}
```



```c++
//AC自动机Trie图优化模版,线性复杂度
//搜索关键词
//给定n个长度不超过50的由小写英文字母组成的单词,以及一篇长为m的文章,请问,有多少个单词在文章中出现了
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>
using namespace std;

const int N=10010,S=55,M=1000010;
int n;
int tr[N*S][26],cnt=[N*S],idx;//cnt表示以每个节点结尾的单词数量
char str[M];
int q[N*S],ne[N*S];//q为bfs队列,ne为next数组

void insert()
{
    int p=0;
    for (int i=0;str[i];i++)
    {
        int t=str[i]-'a';
        if (!tr[p][t]) tr[p][t]=++idx;
        p=tr[p][t];
    }
    cnt[p]++;
}
void build()
{
    int hh=0,tt=-1;
    for (int i=0;i<26;i++)
        if (tr[0][i])
        	q[++tt]=tr[0][i];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=0;i<26;i++)
        {
            int p=tr[t][i];
            if (!p) tr[t][i]=tr[ne[t]][i];
            else
            {
                ne[p]=tr[ne[t]][i];
                q[++tt]=p;
            }
            
        }
    }
}
int main(){
    int T;
    scanf("%d",&T);
    while (T--){
        memset(tr,0,sizeof tr);
        memset(cnt,0,sizeof cnt);
        memset(ne,0,sizeof ne);
        idx=0;
        
        scanf("%d",&n);
        for (int i=0;i<n;i++){
            scanf("%s",str);
            insert();
        }
        
        build();
        scanf("%s",str);
        int res=0;
        for (int i=0,j=0;str[i];i++)
        {
            int t=str[i]-'a';
            j=tr[j][t]
            
            int p=j;
            while (p)
            {
                res+=cnt[p];
                cnt[p]=0;
                p=ne[p];
            }
        }
        printf("%d\n",res);
    }
    return 0;
}
```

# AcWing提高课:RMQ(区间最值查询)算法

RMQ本质是动态规划

先倍增预处理,再查询

即用ST表解决

f(i,j)表示从i开始,长度是$2^j$的区间中,最大值是多少

RMQ不支持修改

时间复杂度:O(nlogn)

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
#include <math>

using namespace std;

const int N=200010,M=18;

int n,m;
int w[N];
int f[N][M];

void init()
{
    for (int j=0;j<M;j++)
        for (int i=1;i+(1<<j)-1<=n;i++)
            if (!j) f[i][j]=w[i];
            else f[i][j]=max(f[i][j-1],f[i+(1<<j-1)][j-1]);
}

int query(int l,int r)
{
    int len=r-l+1;
    int k=log(len)/log(2);
    return max(f[l][k],f[r-(1<<k)+1][k]);
}
int main()
{
    scanf("%d",&n);
    for (int i=1;i<=n;i++) scanf("%d",&w[i]);
    init();
    
    scanf("%d",&m);
    while (m--)
    {
        int l,r;
        scanf("%d%d",&l,&r);
        printf("%d\n",query(l,r));
    }
    return 0;
}
```














































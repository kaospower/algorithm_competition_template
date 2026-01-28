# AcWing进阶课:网络流基本概念

**流网络**:有向图,$G=(V,E)$
源点,汇点,容量,假定任何一条边没有反向边,如果容量不存在,就令c=0
**可行流**:
1.容量限制 0<=f(u,v)<=c(u,v)
2.流量守恒 $\forall x\in V-\{s,t\},除了源点和汇点,其他所有点都满足\displaystyle\sum_{(v,x)\in E}f(v,x)=\displaystyle\sum_{(x,v)\in E}f(x,v)$

注意:不考虑反向边

可行流的流量值|f|=从源点流出的流量-流入源点的流量,即从源点往外净流出的流量:$\displaystyle\sum_{(s,v)\in E}f(s,v)-\displaystyle\sum_{(v,s)\in E}f(v,s)$

**最大流**:最大可行流的流量

一个流网络可能有很多可行流,因为流量不一定是整数

**残留网络**:针对不同的可行流,有不同的残留网络,残留网络和可行流一一对应
$$
G
\begin{cases}f_1\rightarrow G_{f_1}\\
f_2\rightarrow G_{f_2}
\end{cases}
$$
$对于残留网络G_f,V_f=V,E_f=E和E中所有的反向边,即点集和原图相同,边集是原图的两倍$

残留网络的容量:
$$
c'(u,v)=
\begin{cases}
c(u,v)-f(u,v)\quad (u,v)\in E,对于原图的正向边\\
f(v,u)\quad (v,u)\in E,对于原图的反向边
\end{cases}
$$
原网络G的一个可行流f+残留网络中的可行流f'也是原网络G中的一个可行流,|f+f'|=|f|+|f'|

流量相加:每条边对应相加,如果残留网络边的方向和原网络边的方向相同,就累加到原网络中,如果相反,就从原网络中减去

在残留网络中求得的可行流,都可以增加原网络中的可行流

推论:如果残留网络中有可行流,原网络中的可行流就一定不是最大流

残留网络中如果没有可行流,原网络中的可行流就一定是最大流

**增广路径**:在残留网络中,从源点出发,沿着容量大于0的边,如果能够走到终点,就称为增广路径

增广路径一定是可行流,且流量大于0,即最简单的流量大于0的可行流

如果对于当前可行流f,在它的残留网络$G_f$里,找不到增广路径,就说明f是最大流

$G\rightarrow f\rightarrow G_f$

**割**:将流网络G=(V,E)的点集V分成两个不重叠的子集,S和T,满足$S\cup T=V,S\cap T=\varnothing,满足源点s\in S,汇点t\in T$
**割的容量**:所有从S指向T的边的容量之和,$c(S,T)=\displaystyle\sum_{u\in S}\sum_{v\in T}c(u,v)$

**最小割**:最小割的容量

**割的流量**:所有从S流向T的流量-所有从T流向S的流量,$f(S,T)=\displaystyle\sum_{u\in S}\sum_{v\in T}f(u,v)-\sum_{u\in T}\sum_{v\in S}f(u,v)$

$割的流量\leq 割的容量:\forall [S,T],\forall f,f(S,T)\leq c(S,T)$

$割的流量=可行流的流量:f(S,T)=|f|$

即$|f|=f(S,T)\leq c(S,T)$

即$|f|\leq c(S,T)$

即**最大流$\leq$最小割**

**最大流最小割定理**
以下三个条件等价:
1.可行流f是最大流
2.可行流f的残留网络$G_f$中不存在增广路径
3.$\exists割[S,T],|f|=c(S,T)$



**FF**方法:求解最大流的思想

1.在残留网络中利用bfs迭代寻找增广路f'

2.更新残留网络,$G_f=G_{f+f'}$,正向边减去k,反向边加上k,k是增广路每条边容量的最小值,即流量

当找不到增广路时,就找到了最大流



**EK和dinic**是FF的具体实现

EK算法:$O(nm^2)$,每次增广一条路,一般不用它求最大流,但是求费用流时,EK算法十分重要
dinic算法:$O(n^2m)$,每次增广多条路,引入了分层图的概念,dinic和ISAP在求最大流时十分常用

还有如HLPP算法等最大流算法

最大流时间复杂度上界非常宽松,实际复杂度是达不到这个上界的

# AcWing进阶课:最大流

**EK算法**,$O(nm^2)$
存图:邻接表
加边时正向边和反向边成对存储,即第i条边的反向边=i^1,找反向边非常容易

最大流建图正确性:
流网络中的任何一个可行流f都是原问题的一组可行解
原问题的任何一组可行解都是流网络的一个可行流
证明了这两点,原问题的最大值就是流网络的最大流

```c++
//EK算法模版
#include <iostream>
#include <cstring>
#inlcude <algorithm>

using namespace std;

const N=1010,M=20010,INF=1e8;
    
int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;//f表示容量
int q[N],d[N],pre[N];//q:队列,d:从起点走到某个点,容量的最小值,pre:前驱点
bool st[N];//bfs判重数组

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++; //正向边容量是c
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++; //反向边容量是0
}

//宽搜找增广路
bool bfs()
{
    int hh=0,tt=0;
    memset(st,false,sizeof st);
    q[0]=S,st[S]=true,d[S]=INF; //起点加入队列,一开始没有任何限制,所以d[S]=INF
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (!st[ver] && f[i]) //f[i]要大于0,根据增广路定义,增广路上所有边容量都要大于0
            {
                st[ver]=true;
                d[ver]=min(d[t],f[i]);
                
                pre[ver]=i;//前驱边
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}
int EK()
{
    int r=0;
    //迭代更新残留网络
    while (bfs())
    {
        r+=d[T];
        for (int i=T;i!=S;i=e[pre[i]^1])
            f[pre[i]]-=d[T],f[pre[i]^1]+=d[T];
    }
    return r;
}
int main()
{
    scanf("%d%d%d%d",&n,&m,&S,&T);
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c);
    }
    printf("%d\n",EK());
    return 0;
}
```

**Dinic算法**,$O(n^2m)$在求解最大流最小割问题时更常用,EK算法在求解这些问题时基本不会用

为了处理环的问题,dinic算法引入了分层图的概念,在寻找路径时,每次只能从前一层走到后一层,从而保证没有环路

1.bfs建立分层图

2.dfs找出所有能够增广路径,无回溯

**当前弧优化**:如果某条边已经满了,下次搜索的时候就跳过这条边,从它的下一条边开始搜

代码中维护的图是残留网络$G_f$

```c++
//dinic算法模版
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=10010,M=200010,INF=1e8;

int n,m,S,T;//S:源点,T:汇点
int h[N],e[M],f[M],ne[M],idx;//f表示残留网络容量
int q[N],d[N],cur[N];//q:队列,d:分层图层数,cur:当前弧优化

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;//正向边
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;//反向边
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];//起点入队,cur表示当前弧优化
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
    
}

//深搜
int find(int u,int limit)//从u开始搜,从起点流向u的最大流量是limit
{
    if (u==T) return limit;
    int flow=0;//flow表示从这个点往后流的流量最多是多少
    //从当前没有满的路径开始搜,flow<limit是重要优化,如果不加这个条件会TLE
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i; //当前弧优化
        int ver=e[i];
        //搜索下一层
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;//删点
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    //bfs会返回是否存在增广路,如果有增广路会同时建出分层图
    //find()是dfs函数
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d%d%d",&n,&m,&S,&T);
    memeset(h,-1,sizeof h);
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c); //每条边的起点,终点,容量
        add(a,b,c);
    }
    printf("%d\n",dinic());
    return 0;
}
```

**最大流问题分析方式**:

可行解的集合和可行流的集合中的元素一一对应,所以可行解的最大值等于可行流的最大值

即最优解=最大流

**最大流之二分图匹配**

匈牙利算法解决二分图匹配时间复杂度是$O(nm)$

dinic算法解决二分图匹配时间复杂度是$O(m\sqrt{n})$

dinic算法更快

建出虚拟源点S和汇点T

从源点向第一个点集的每一个点连一条容量是1的边

从第二个点集向汇点连一条容量是1的边

对于两个点集之间的边,容量是1

```c++
//飞行员配对方案问题
//最大流解决二分图匹配问题模版
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=110,M=5210,INF=1e8;

int m,n,S,T;
int h[n],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}
int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=h[u];~i && flow<limit;i=ne[i])
    {
        cur[i]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i]){
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow==find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d",&m,&n);
    S=0,T=n+1;
    memset(h,-1,sizeof h);
    for (int i=1;i<=m;i++) add(S,i,1);
    for (int i=m+1;i<=n;i++) add(i,T,1);
    
    int a,b;
    while (cin>>a>>b,a!=-1) add(a,b,1);
    printf("%d\n",dinic());
    for (int i=0;i<idx;i+=2) //枚举所有正向边
        if (e[i]>m && e[i]<=n && !f[i])
            printf("%d %d\n",e[i^1],e[i]);
    return 0;
}
```



```c++
//圆桌问题
//最大流解决二分图多重匹配问题
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=430,M=(150*270+N)*2,INF=1e8;

int m,n,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}
int find(int u,int limit)
{
    if(u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d",&m,&n);
    S=0,T=m+n+1;
    memeset(h,-1,sizeof h);
    int tot=0;
    for (int i=1;i<=m;i++)
    {
        int c;
        scanf("%d",&c);//c是人数
        add(S,i,c);
        tot+=c;
    }
    for (int i=1;i<=n;i++)
    {
        int c;
        scanf("%d",&c);
        add(m+i,T,c);
    }
    for (int i=1;i<=m;i++)
        for (int j=1;j<=n;j++)
            add(i,m+j,1);
    if (dinic()!=tot) puts("0");
    else {
        puts("1");
        for (int i=1;i<=m;i++)
        {
            for (int j=h[i];~j;j=ne[j])
                if (e[j]>m && e[j]<=m+n && !f[j])
                    printf("%d",e[j]-m);
            puts("");
        }
    }
    return 0;
}
```

**无源汇上下界可行流**

容量多了一个下界限制
$$
\begin{equation*}
\begin{cases}
流量守恒\\
容量限制:c_l(u,v)\leq f(u,v)\leq c_u(u,v)
\end{cases}
\end{equation*}
$$
对上述容量限制进行变形:

$0\leq f(u,v)-c_l(u,v)\leq c_u(u,v)-c_l(u,v)$

令f'的容量为$c_u(u,v)-c_l(u,v)$

源点和汇点补充流量使得新图满足流量守恒

```c++
//无源汇上下界可行流模版
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=210,M=(10200+N)*2,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],l[M],ne[M],idx;//l代表容量下界
int q[N],d[N],cur[N],A[N];//A[i]表示对于点i,进入的容量下界之和-出去的容量下界之和

void add(a,b,c,d)
{
    e[idx]=b,f[idx]=d-c,l[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;//反向边下界用不到
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}
int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d",&n,&m);
    S=0,T=n+1;
    memset(h,-1,sizeof h);
    for (int i=0;i<m;i++)
    {
        int a,b,c,d;
        scanf("%d%d%d%d",&a,&b,&c,&d);//每条边的起点,终点,容量上界,容量下界
        add(a,b,c,d);
        A[a]-=c,A[b]+=c;
    }
    int tot=0;//从源点出发到每个点的容量之和
    for (int i=1;i<=n;i++)
        if (A[i]>0) add(S,i,0,A[i]),tot+=A[i];
    	else if (A[i]<0) add(i,T,0,-A[i]);
    if (dinic()!=tot) puts("NO"); //如果不是满流
    else
    {
        puts("YES");
        for (int i=0;i<m*2;i+=2)
            printf("%d\n",f[i^1]+l[i]);
    }
    return 0;
}
```

**有源汇上下界最大流**

从汇点连一条到源点的容量是正无穷的边,转化成无源汇问题,同时使源点和汇点满足流量守恒

```c++
//有源汇上下界最大流模版
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=210,M=(N+10000)*2,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N],A[N];

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]=-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    int s,t;//原始源点,汇点
    scanf("%d%d%d%d",&n,&m,&s,&t);
    S=0,T=n+1;
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b,c,d;
        scanf("%d%d%d%d",&a,&b,&c,&d);
        add(a,b,d-c);
        A[a]-=c,A[b]+=c;
    }
    int tot=0;
    for (int i=1;i<=n;i++)
        if (A[i]>0) add(S,i,A[i]),tot+=A[i];
    	else if (A[i]<0) add(i,T,-A[i]);
    add(t,s,INF);
    if (dinic()<tot) puts("No Solution");
    else
    {
        int res=f[idx-1];
        S=s,T=t;
        f[idx-1]=f[idx-2]=0;
        printf("%d\n",res+dinic());
    }
    return 0;
}
```

**有源汇上下界最小流**

```c++
//有源汇上下界最小流模版
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=50010,M=(N+125003)*2,INF=2147483647;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N],A[N];

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}
int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    int s,t;
    scanf("%d%d%d%d",&n,&m,&s,&t);
    S=0,T=n+1;
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b,c,d;
        scanf("%d%d%d%d",&a,&b,&c,&d);
        add(a,b,d-c);
        A[a]-=c,A[b]+=c;
    }
    int tot=0;
    for (int i=1;i<=n;i++)
        if (A[i]>0) add(S,i,A[i]),tot+=A[i];
    	else if (A[i]<0) add(i,T,-A[i]);
    add(t,s,INF);
    if (dinic()<tot) puts("No Solution");
    else
    {
        int res=f[idx-1];
        S=t,T=s;
        f[idx-1]=f[idx-2]=0;
        printf("%d\n",res-dinic());
    }
    return 0;
}
```

**多源汇最大流**

建立虚拟源点和汇点,从虚拟源点向所有原始源点分别连一条容量是正无穷的边,从所有原始汇点分别向虚拟汇点连一条容量是正无穷的边,虚拟源点到虚拟汇点的最大流即为原图最大流

```c++
//多源汇最大流模版
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=10010,M=(100000+N)*2,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}
int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t
        } 
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}

int main()
{
    int sc,tc;
    scanf("%d%d%d%d",&n,&m,&sc,&tc);
    S=0,T=n+1;
    memset(h,-1,sizeof h);
    while (sc--)
    {
        int x;
        scanf("%d",&x);
        add(S,x,INF);
    }
    while (tc--)
    {
        int x;
        scanf("%d",&x);
        add(x,T,INF);
    }
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d%d",&a,&b,&c);
        add(a,b,c);
    }
    printf("%d\n",dinic());
    return 0;
}
```



```c++
//伊基的故事I-道路重建
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=510,M=10010,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
bool vis_s[N],vis_t[N];//vis_s表示所有从源点能到的点,vis_t表示所有能到汇点的点

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]=d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
void dfs(int u,bool st[],int t)
{
    st[u]=true;
    for (int i=h[u];~i;i=ne[i])
    {
        int j=i^t,,ver=e[i];
        if (f[j] && !st[ver])
            dfs(ver,st,t);
    }
}
int main()
{
    scanf("%d%d",&n,&m);
    S=0,T=n-1;
    memset(h,-1,sizeof h);
    for (int i=0;i<m;i++)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c);
    }
    dinic();
    dfs(S,vis_s,0);
    dfs(T,vis_t,1);
    int res=0;
    for (int i=0;i<m*2;i+=2)
        if (!f[i] && vis_s[e[i^1]] && vis_t[e[i]]) //!f[i]表示满流
            res++;
    printf("%d\n",res);
    return 0;
}
```

**最大流之判定**

```c++
//二分出边界
//秘密挤奶机

#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=210,M=80010,INF=1e8;

int n,m,K,S,T;
int h[N],e[M],f[M],w[M],ne[M],idx;
int q[N],d[N],cur[N];

void add(int a,int b,int c)
{
    e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,w[idx]=c,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
bool check(int mid)
{
    for (int i=0;i<idx;i++)
        if (w[i]>mid) f[i]=0;
    	else f[i]=1;
    return dinic()>=K;
}
int main()
{
    scanf("%d%d%d",&n,&m,&K);
    S=1,T=N;
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c);
    }
    int l=1,r=1e6;
    while (l<r)
    {
        int mid=l+r>>1;
        if (check(mid)) r=mid;
        else l=mid+1;
    }
    printf("%d\n",r);
    return 0;
}
```



```c++
//星际转移问题
//用并查集判断0和n+1是否连通
//分层图
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=1101*22+10,M=(N+1100+13*1101)*2+10,INF=1e8;
int n,m,k,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
struct Ship
{
    int h,r,id[30];
}ships[30];
int p[30];//并查集

int find(int x)
{
    if (p[x]!=x) p[x]=find(p[x]);
    return p[x];
}
int get(int i,int day)
{
    return day*(n+2)+i;
}
void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}
int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d%d",&n,&m,&k);
    S=N-2,T=N-1;
    memset(h,-1,sizeof h);
    for (int i=0;i<30;i++) p[i]=i;
    for (int i=0;i<m;i++)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        ships[i]={a,b};
        for (int j=0;j<b;j++)
        {
            int id;
            scanf("%d",&id);
            if (id==-1) id=n+1;
            ships[i].id[j]=id;
            if (j)
            {
                int x=ships[i].id[j-1];
                p[find(x)]=find(id);
            }
        }
    }
    if (find(0)!=find(n+1)) puts("0");
    else 
    {
       add(S,get(0,0),k); 
       add(get(n+1,0),T,INF);
       int day=1,res=0;
       while (true)
       {
           add(get(n+1,day),T,INF);
           for (int i=0;i<=n+1;i++)
               add(get(i,day-1),get(i,day),INF);
           for (int i=0;i<m;i++)
           {
               int r=ships[i].r;
               int a=ships[i].id[(day-1)%r],b=ships[i].id[day%r];
               add(get(a,day-1),get(b,day),ships[i].h);
           }
           res+=dinic();
           if (res>=k) break;
           day++;
       }
       printf("%d\n",day);
    }
    return 0;
}
```

**最大流拆点**

拆点,处理匹配问题时,将一个点拆成入点和出点

```c++
//三分图匹配
//餐饮
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=410,M=40610,INF=1e8;

int n,F,D,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}

int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d%d",&n,&F,&D);
    S=0,T=n*2+F+D+1;
    memset(h,-1,sizeof h);
    for (int i=1;i<=F;i++) add(S,n*2+i,1);
    for (int i=1;i<=D;i++) add(n*2+F+i,T,1);
    for (int i=1;i<=n;i++)
    {
        add(i,n+i,1);
        int a,b;
        scanf("%d%d",&a,&b);
        while (a--)
        {
            scanf("%d",&t);
            add(n*2+t,i,1);
        }
        while (b--)
        {
            scanf("%d",&t);
            add(i+n,n*2+F+t,1);
        }
    }
    printf("%d\n",dinic());
    return 0;
}
```



```c++
//最长递增子序列问题
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=1010,M=251010,INF=1e8;

int n,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
int g[N],w[N];
void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}
int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d",&n);
    S=0,T=n*2+1;
    memset(h,-1,sizeof h);
    for (int i=1;i<=n;i++) scanf("%d",&w[i]);
    int s=0;
    for (int i=1;i<=n;i++)
    {
        add(i,i+n,1);
        g[i]=1;
        for (int j=1;j<i;j++)
            if (w[j]<=w[i])
                g[i]=max(g[i],g[j]+1);
        for (int j=1;j<i;j++)
            if (w[j]<=w[i] && g[j]+1==g[i])
                add(n+j,i,1);
        s=max(s,g[i]);
        if (g[i]==1) add(S,i,1);
    }
    for (int i=1;i<=n;i++)
        if (g[i]==S)
            add(n+i,T,1);
    printf("%d\n",s);
    if (s==1) printf("%d\n%d\n",n,n);
    else 
    {
        int res=dinic();
        printf("%d\n",res);
        for (int i=0;i<idx;i+=2)
        {
            int a=e[i^1],b=e[i];
            if (a==S && b==1) f[i]=INF;
            else if (a==1 && b==n+1) f[i]=INF;
            else if (a==n && b==n+n) f[i]=INF;
            else if (a==n+n && b==T) f[i]=INF;
        }
        printf("%d\n",res+dinic());
    }
    return 0;
}
```



```c++
//企鹅旅行
#include <iostream>
#include <algorithm>
#include <cstring>

#define x first;
#define y second;
using namespace std;

typedef pair<int,int>PII;
const int N=210,M=20410,INF=1e8;
const double eps=1e-8;

int n,S,T;
double D;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
PII p[N];

bool check(PII a,PII b)
{
    double dx=a.x-b.x,dy=a.y-b.y;
    return dx*dx+dy*dy<=D*D+eps;
}

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}
int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    int cases;
    scanf("%d",&cases);
    while (cases--)
    {
        memset(h,-1,sizeof h);
        idx=0;
        scanf("%d%lf",&n,&D);
        S=0;
        int tot=0;
        for (int i=1;i<=n;i++)
        {
            int x,y,a,b;
            scanf("%d%d%d%d",&x,&y,&a,&b);
            p[i]={x,y};
            add(S,i,a);
            add(i,n+i,b);
            tot+=a;
        }
        for (int i=1;i<=n;i++)
            for (int j=i+1;j<=n;j++)
                if (check(p[i],q[j]))
                {
                    add(n+i,j,INF);
                    add(n+j,i,INF);
                }
        int cnt=0;
        for (int i=1;i<=n;i++)
        {
            T=i;
            for (int j=0;j<idx;j+=2)
            {
                f[j]+=f[j^1];
                f[j^1]=0;
            }
            if (dinic()==tot)
            {
                printf("%d",i-1);
                cnt++;
            }
        }
        if (!cnt) puts("-1");
        else puts("");
        
    }
    return 0;
}
```



```c++
//猪
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=110,M=20210,INF=1e8;

int m,n,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
int w[1010],last[1010];

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}
int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d",&m,&n);
    S=0,T=n+1;
    memset(h,-1,sizeof h);
    for (int i=1;i<=m;i++) scanf("%d",&w[i]);
    for (int i=1;i<=n;i++)
    {
        int a,b;
        scanf("%d",&a);
        while (a--)
        {
            int id;
            scanf("%d",&id);
            if (!last[id]) add(S,i,w[id]);
            else add(last[id],i,INF);
            last[id]=i;
        }
        scanf("%d",&b);
        add(i,T,b);
    }
    printf("%d\n",dinic());
    return 0;
}
```



# AcWing进阶课:最小割

最大流=最小割

```c++
//dinic求最小割
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=10010,M=200010,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}

int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d%d%d",&n,&m,&S,&T);
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c);
    }
    printf("%d\n",dinic());
    return 0;
}
```



**01分数规划**



```c++
//网络战争

#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=110,M=810,INF=1e8;
const double eps=1e-8;

int n,m,S,T;
int h[N],e[M],w[M],ne[M],idx;
double f[m];
int q[N],d[N],cur[N];

void add(int a,int b,int c)
{
    e[idx]=b,w[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,w[idx]=c,ne[idx]=h[b],h[b]=idx++;
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

double find(int u,double limit)
{
    if (u==T) return limit;
    double flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            double t=find(ver,min(f[i],limit-flow));
            if (t<eps) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
double dinic(double mid)
{
	double res=0;
    for (int i=0;i<idx;i+=2)
        if (w[i]<=mid)
        {
            res+=w[i]-mid;
            f[i]=f[i^1]=0;
        }
    	else f[i]=f[i^1]=w[i]-mid;
    double r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r+res;
}

int main()
{
    scanf("%d%d%d%d",&n,&m,&S,&T);
    memset(h,-1,sizeof h);
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c);
    }
    
    double l=0,r=1e7;
    while (r-l>eps)
    {
        double mid=(l+r)/2;
        if (dinic(mid)<0) r=mid;
        else l=mid;
    }
    printf("%.2lf\n",r);
    return 0;
}
```



```c++
//最优标号
#include <iostream>
#include <algorithm>
#include <cstring>

#define x first;
#define y second;
using namespace std;

typedef long long LL;
typedef pair<int,int>PII;
const int N=510,M=(3000+N*2)*2,INF=1e8;

int n,m,k,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
int p[N];
PII edges[3010];

void add(int a,int b,int c1,int c2)//c1表示正向边容量,c2表示反向边容量
{
    e[idx]=b,f[idx]=c1,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=c2,ne[idx]=h[b],h[b]=idx++;
    
}
void build(int k)
{
    memset(h,-1,sizeof h);
    idx=0;
    for (int i=0;i<m;i++)
    {
        int a=edges[i].x,b=edges[i].y;
        add(a,b,1,1);
    }
    for (int i=1;i<=n;i++)
        if (p[i]>=0)
        {
            if (p[i]>>k&1) add(i,T,INF,0);
            else add(S,i,INF,0);
        }
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
        	int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
LL dinic(int k)
{
    build(k);
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof h);
    S=0,T=n+1;
    for (int i=0;i<m;i++) scanf("%d%d",&edges[i].x,&edges[i].y);
    scanf("%d",&k);
    memset(p,-1,sizeof p);
    while (k--)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        p[a]=b;
    }
    
    LL res=0;
    for (int i=0;i<=30;i++) res+=dinic()<<i;
    printf("%lld\n",res);
    return 0;
}
```



**最大权闭合子图**

**闭合子图**:对于一个有向图的点的集合,点集内部的点不能从点集里面指向点集外面

**简单割**:所有割边都和源点,汇点相连

最小割一定是简单割

$闭合子图\Leftrightarrow 简单割$

```txt
构造流网络:
从源点向所有权值为正数的点连一条边,从所有权值为负数的点向汇点连一条边,新连的边的容量取绝对值,原图的边的容量设为正无穷
最大权闭合子图的权=所有正数点权之和-最小割
```



```c++
//最大获利
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=55010,M=(50000*3+5000)*2+10,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}

int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}

int main(){
    scanf("%d%d",&n,&m);
    S=0,T=n+m+1;
    memset(h,-1,sizeof h);
    for (int i=1;i<=n;i++)
    {
        int p;
        scanf("%d",&p);
        add(m+i,T,p);
    }
    int tot=0;
    for (int i=1;i<=m;i++)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(S,i,c);
        add(i,m+a,INF);
        add(i,m+b,INF);
        tot+=c;
    }
    printf("%d\n",tot-dinic());
    return 0;
}
```



```c++
//利用最大密度子图优化该题
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=5010,M=(50000+N*2)*2+10,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
int dg[N],p[N];

void add(int a,int b,int c1,int c2)
{
    e[idx]=b,f[idx]=c1,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=c2,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}

int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d",&n,&m);
    S=0,T=n+1;
    memset(h,-1,sizeof h);
    for (int i=1;i<=n;i++) scanf("%d",&p[i]),p[i]*=-1;
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c,c);
        dg[a]+=c,dg[b]+=c;
    }
    int U=0;
    for (int i=1;i<=n;i++) U=max(U,2*p[i]+dg[i]);
    for (int i=1;i<=n;i++)
    {
        add(S,i,U,0);
        add(i,T,U-2*p[i]-dg[i],0);
    }
    printf("%d\n",(U*n-dinic())/2);
    return 0;
}
```



**最大密度子图问题**

选边就必须选点,最大化密度$\frac{E'}{|V'|}$

将原图的边看成点

```c++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N=110,M=(1000+N*2)*2,INF=1e8;

int n,m,S,T;
int h[N],e[M],ne[M],idx;
double f[M];
int q[N],d[N],cur[N];
int dg[N];
int ans;
bool st[N];

struct Edge
{
    int a,b;
}edges[M];

void add(int a,int b,double c1,double c2)
{
    e[idx]=b,f[idx]=c1,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=c2,ne[idx]=h[b],h[b]=idx++;
}
void build (double g)
{
    memset(h,-1,sizeof h);
    idx=0;
    for (int i=0;i<m;i++) add(edges[i].a,edges[i].b,1,1);
    for (int i=1;i<n;i++)
    {
        add(S,i,m,0);
        add(i,T,m+g*2-dg[i],0);
    }
}

bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

double find(int u,double limit)
{
    if (u==T) return limit;
    double flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            double t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}
double dinic(double g)
{
    build(g);
    double r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}

void dfs(int u)
{
    st[u]=true;
    if (u!=S) ans++;
    for (int i=h[u];~i;i=ne[i])
    {
        int ver=e[i];
        if (!st[ver] && f[i])
            dfs(ver);
    }
}
int main()
{
    scanf("%d%d",&n,%m);
    S=0,T=n+1;
    for (int i=0;i<m;i++)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        dg[a]++,dg[b]++;
        edges[i]={a,b};
    }
    double l=0,r=m;
    while (r-1>1e-8)
    {
        double mid=(l+r)/2;
        double t=dinic(mid);
        if (m*n-t>0) l=mid;
        else r=mid;
    }
    dinic(l);
    dfs(S);
    if (!ans) puts("1\n1");
    else
    {
        printf("%d\n",ans);
        for (int i=1;i<=n;i++)
            if (st[i])
        		printf("%d\n",i);
    
    }
    return 0;
}
```



**最小权点覆盖集**:

**点覆盖集**:从无向图中选一些点,使得每一条边都至少有一个点被选出来

**最小权点覆盖集**:图中每一个点都是非负权值,在所有点覆盖集中选出一个权值最小的点覆盖集

对于一般图来说:最小权点覆盖集问题是NP完全问题,只能用暴搜解决

二分图的最小权点覆盖集可以用特殊方法解决

如果点权是1:二分图的最大匹配数=最小点覆盖数=n-最大独立集数,可以用匈牙利算法解决

如果点权不是1:只能用网络流来解决

当我们期望最小割取不到某些边时,就将这些边的容量设为正无穷

简单割的容量就是所有割边的容量之和

```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=210,M=5200*2+10,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
bool st[N];

void dfs(int u)
{
    st[u]=true;
    for (int i=h[u];~i;i=ne[i])
        if (f[i] && !st[e[i]])
            dfs(e[i]);
}

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}

int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d",&n,&m);
    S=0,T=n*2+1;
   	memset(h,-1,sizeof h);
    for (int i=1;i<=n;i++)
    {
        int w;
        scanf("%d",&w);
        add(S,i,w);
    }
    for (int i=1;i<=n;i++)
    {
        int w;
        scanf("%d",&w);
        add(n+i,T,w);
        
    }
    while (m--)
    {
        int a,b;
        scanf("%d%td",&a,&b);
        add(b,n+a,INF);
    }
    printf("%d\n",dinic());
    dfs(S);
    
    int cnt=0;
    for (int i=0;i<idx;i+=2)
    {
        int a=e[i^1],b=e[i];
        if (st[a] && !st[b]) cnt++;
    }
    printf("%d\n",cnt);
    for (int i=0;i<idx;i+=2)
    {
        int a=e[i^1],b=e[i];
        if (st[a] && !st[b])
        {
            if (a==S) printf("%d +\n",b);
        }
    }
    for (int i=0;i<idx;i+=2)
    {
        int a=e[i^1],b=e[i];
        if (st[a] && !st[b])
        {
            if (b==T) printf("%d -\n",a-n);
        }
    }
    return 0;
}
```



**最大权独立集**

二分图权值是1时,最大独立集=n-最小覆盖集

二分图权值不为1时,最大权独立集=所有点的总权值 - 最小权点覆盖

**独立集**:无向图中选某些点,使得选出的所有点之间都没有边,这样的点集称为独立集

**最大权独立集**:所有点权都是非负,在所有的独立集中,点权和最大的独立集

一般图的最大权独立集问题是NP完全问题,只能用暴搜解决

二分图的最大权独立集问题可以用特殊方法解决

```c++
//王者之剑
//只能在偶数秒拿宝石,不能同时拿走相邻格子上的宝石,即独立集
#include<iostream>
#include<algorithm>
#include<cstring>

using namespace std;

const int N=10010,M=60010,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];

int get(int x,int y)
{
    return (x-1)*m+y;
}
void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}

int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}

int main()
{
    scanf("%d%d",&n,&m);
    S=0,T=n*m+1;
    memset(h,-1,sizeof h);
    int dx[]={-1,0,1,0},dy[]={0,1,0,-1};
    
    int tot=0;
    for (int i=1;i<=n;i++)
        for (int j=1;j<=m;j++)
        {
            int w;
            scanf("%d",&w);
            if (i+j &1)
            {
                add(S,get(i,j),w);
                for (int k=0;i<4;k++)
                {
                    int x=i+dx[k],y=j+dy[k];
                    if (x>=1 && x<=n && y>=1 && y<=m)
                        add(get(i,j),get(x,y),INF);
                }
                    
            }
            else
                add(get(i,j),T,w);
            tot+=w;
        }
    printf("%d\n",tot-dinic());
    return 0;
}
```



```c++
//有线电视网络
//给定一张n个点m条边的无向图,求最少去掉多少个点,可以使图不连通
//枚举源点和汇点,拆点,将所有内部边容量设为1,外部边容量设为正无穷,构造简单割,答案即为流网络的最小割
#include<iostream>
#include<algorithm>
#include<cstring>

using namespace std;

const int N=110,M=5210,INF=1e8;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
int n,m,S,T;
void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}

int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    while (cin>>n>>m)
    {
        memset(h,-1,sizeof h);
        idx=0;
        for (int i=0;i<n;i++) add(i,n+i,1);
        while(m--)
        {
            int a,b;
            scanf(" (%d,%d)",&a,&b);
            add(n+a,b,INF);
            add(n+b,a,INF);
        }
        int res=n;
        for (int i=0;i<n;i++)
            for (int j=0;j<i;j++)
            {
                S=n+i,T=j;
                for (int k=0;k<idx;k+=2)
                {
                    f[k]+=f[k^1];
                    f[k^1]=0;
                }
                res=min(res,dinic());
            }
        printf("%d\n",res);
    }
    return 0;
}
```



```c++
//太空飞行计划
//最大权闭合图问题
#include<iostream>
#include<algorithm>
#include<sstream>
#include<cstring>

using namespace std;

const int N=110,M=5210,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
bool st[N];

void dfs(int u)
{
    st[u]=true;
    for (int i=h[u];~i;i=ne[i])
        if (!st[e[i]] && f[i])
            dfs(e[i]);
}

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}

int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d",&m,&n);
    S=0,T=m+n+1;
    memset(h,-1,sizeof h);
    gechar();
    
    int tot=0;
    for (int i=1;i<=m;i++)
    {
        int w,id;
        string line;
        getline(cin,line);
        stringstream ssin(line);
        ssin>>w;
        add(S,i,w);
        while (ssin>>id) add(i,m+id,INF);
        tot+=w;
    }
    for (int i=1;i<=n;i++)
    {
        int p;
        cin>>p;
        add(m+i,T,p);
    }
    int res=dinic();
    dfs(S);
	for (int i=1;i<=m;i++)
        if (st[i]) printf("%d ",i);
    puts("");
    for (int i=m+1;i<=m+n;i++)
        if (st[i]) printf("%d ",i-m);
    puts("");
    printf("%d\n",tot-res);
    return 0;
}
```



```c++
//此题是二分图的最大权独立集模型
//棋盘上放马,使得不能相互攻击,同时棋盘上有障碍
#include<iostream>
#include<algorithm>
#include<cstring>

using namespace std;

const int N=40010,M=400010,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],ne[M],idx;
int q[N],d[N],cur[N];
bool g[210][210];//记录障碍物

int get(int x,int y)
{
    return (x-1)*n+y;
}

void add(int a,int b,int c)
{
    e[idx]=b,f[idx]=c,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,ne[idx]=h[b],h[b]=idx++;
}
bool bfs()
{
    int hh=0,tt=0;
    memset(d,-1,sizeof d);
    q[0]=S,d[S]=0,cur[S]=h[S];
    while (hh<=tt)
    {
        int t=q[hh++];
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (d[ver]==-1 && f[i])
            {
                d[ver]=d[t]+1;
                cur[ver]=h[ver];
                if (ver==T) return true;
                q[++tt]=ver;
            }
        }
    }
    return false;
}

int find(int u,int limit)
{
    if (u==T) return limit;
    int flow=0;
    for (int i=cur[u];~i && flow<limit;i=ne[i])
    {
        cur[u]=i;
        int ver=e[i];
        if (d[ver]==d[u]+1 && f[i])
        {
            int t=find(ver,min(f[i],limit-flow));
            if (!t) d[ver]=-1;
            f[i]-=t,f[i^1]+=t,flow+=t;
        }
    }
    return flow;
}

int dinic()
{
    int r=0,flow;
    while (bfs()) while (flow=find(S,INF)) r+=flow;
    return r;
}
int main()
{
    scanf("%d%d",&n,&m);
    S=0,T=n*n+1;
    memset(h,-1,sizeof h);
    while (m--)
    {
        int x,y;
        scanf("%d%d",&x,&y);
        g[x][y]=true;
        
    }
    int dx[]={-2,-1,1,2,2,1,-1,-2};
    int dy[]={1,2,2,1,-1,-2,-2,-1};
    
    int tot=0;
    for (int i=1;i<=n;i++)
        for (int j=1;j<=n;j++)
        {
            if (g[i][j]) continue;
            if (i+j &1)
            {
                add(S,get(i,j),1);
                for (int k=0;k<8;k++)
                {
                    int x=i+dx[k],y=j+dy[k];
                    if (x>=1 && x<=n && y>=1 && y<=n && !g[x][y])
                        add(get(i,j),get(x,y),INF);
                }
            }
            else add(get(i,j),T,1);
            tot++;
                
        }
    printf("%d\n",tot-dinic());
    return 0;
}
```



# AcWing进阶课:费用流

**费用流**:所有最大可行流中费用的最小/最大值,简称为最小/最大费用最大流

可行流的总费用=流量$\times$费用

求最小费用最大流,将EK算法中的bfs换成spfa

该算法不能处理有负权回路的网络,因为spfa不能处理负环,如果想处理负环,需要用到消圈法模版

w(v,u)=-w(u,v)即正向边的费用和反向边的费用相反



EK算法:

1.在残留网络中找一条增广路

2.更新残留网络$G_f$

求费用流时将bfs替换成spfa,相当于每次迭代不是再求任意一条增广路,而是求从源点到汇点的最短/最长增广路

```c++
//最小费用最大流模版
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=5010,M=100010,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],w[M],ne[M],idx;//w表示每条边的费用
int q[N],d[N],pre[N],incf[N];//pre:从后往前把spfa最短路推出的数组,incf:走到每个点的时候最大流量
bool st[N];//spfa判重数组

void add(int a,int b,int c,int d){
    e[idx]=b,f[idx]=c,w[idx]=d,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,w[idx]=-d,ne[idx]=h[b],h[b]=idx++;
}


bool spfa()
{
    int hh=0,tt=1;
    memset(d,0x3f,sizeof d);//初始化距离
    memset(incf,0,sizeof incf);
    q[0]=S,d[S]=0,incf[S]=INF;
    while (hh!=tt)//循环队列
    {
        int t=q[hh++];
        if (hh==N) hh=0;
        st[t]=false;
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (f[i] && d[ver]>d[t]+w[i])
            {
                d[ver]=d[t]+w[i];
                pre[ver]=i;
                incf[ver]=min(f[i],incf[t]);
                if (!st[ver])
                {
                    q[tt++]=ver;
                    if (tt==N) tt=0;
                    st[ver]=true;
                }
            }
        }
    }
    return incf[T]>0;
}
void EK(int& flow,int& cost){//以引用参数形式返回
    flow=cost=0;
    while (spfa()){
        int t=incf[T];//走到终点时最大流量是多少
        flow+=t,cost+=t*d[T];
        for (int i=T;i!=S;i=e[pre[i]^1])//反向边的终点就是前一个点
        {
            f[pre[i]]-=t;
            f[pre[i]^1]+=t;
        }
    }
}
int main(){
    scanf("%d%d%d%d",&n,&m,&S,&T);
    memset(h,-1,sizeof h);
    while (m--){
        int a,b,c,d;
        scanf("%d%d%d%d",&a,&b,&c,&d);
        add(a,b,c,d);
        
    }
    int flow,cost;
    EK(flow,cost);
    printf("%d %d\n",flow,cost);
    
    return 0;
}
```



```c++
//运输问题
//多源汇网络流问题,建立虚拟源点,虚拟汇点
//最小费用最大流问题
#inlcude <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=160,M=5150*2+10,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],w[M],ne[M],idx;
int q[N],d[N],pre[N],incf[N];//注意f是容量,incf是流量
bool st[N];

void add(int a,int b,int c,int d)
{
    e[idx]=b,f[idx]=c,w[idx]=d,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,w[idx]=-d,ne[idx]=h[b],h[b]=idx++;
}

bool spfa()
{
    int hh=0,tt=1;
    memset(d,0x3f,sizeof d);
    memset(incf,0,sizeof incf);
    q[0]=S,d[S]=0,incf[S]=INF;
    while (hh!=tt)
    {
        int t=q[hh++];
        if (hh==N) hh=0;
        st[t]=false;
        
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (f[i] && d[ver]>d[t]+w[i])
            {
                d[ver]=d[t]+w[i];
                pre[ver]=i;
                incf[ver]=min(incf[t],f[i]);
                if (!st[ver])
                {
                    q[tt++]=ver;
                    if (tt==N) tt=0;
                    st[ver]=true;
                }
            }
        }
    }
    return incf[T]>0;
}
int EK()
{
    int cost=0;
    while (spfa())
    {
        int t=infc[T];
        cost+=t*d[T];
        for (int i=T;i!=S;i=e[pre[i]^1])
        {
            f[pre[i]]-=t;
            f[pre[i]^1]+=t;
        }
    }
    return cost;
}
int main(){
    scanf("%d%d",&m,&n);
    S=0,T=m+n+1;
    memset(h,-1,sizeof h);
    for (int i=1;i<=m;i++)
    {
        int a;
        scanf("%d",&a);
        add(S,i,a,0);
    }
    for (int i=1;i<=n;i++)
    {
        int b;
        scanf("%d",&b);
        add(m+i,T,b,0);
    }
    for (int i=1;i<=m;i++)
        for (int j=1;j<=n;j++)
        {
            int c;
            scanf("%d",&c);
            add(i,m+j,INF,c);
        }
    printf("%d\n",EK());
    for (int i=0;i<idx;i+=2) //最大费用取反费用即可
    {
        f[i]+=f[i^1],f[i^1]=0;
        w[i]=-w[i],w[i^1]=-w[i^1];
    }
    printf("%d\n",-EK());
    
    return 0;         
}
```



```c++
//负载平衡问题
//最大费用最大流
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=110,M=610,INF=1e8;

int n,S,T;
int S[N];
int h[N],e[M],f[M],w[M],ne[M],idx;
int q[N],d[N],pre[N],incf[N];
bool st[N];

void add(int a,int b,int c,int d)
{
    e[idx]=b,f[idx]=c,w[idx]=d,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,w[idx]=-d,ne[idx]=h[b],h[b]=idx++;
}
bool spfa()
{
    int hh=0,tt=1;
    memset(d,-0x3f,sizeof d);
    memset(incf,0,sizeof incf);
    q[0]=S,d[S]=0,incf[S]=INF;
    while (hh!=tt)
    {
        int t=q[hh++];
        if (hh==N) hh=0;
        st[t]=false;
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (f[i] && d[ver]>d[t]+w[i])
            {
                d[ver]=d[t]+w[i];
                pre[ver]=i;
                incf[ver]=min(f[i],incf[t]);
                if (!st[ver])
                {
                    q[tt++]=ver;
                    if (tt==NN) tt=0;
                    st[ver]=true;
                }
            }
        }
    }
    return incf[T]>0;
}
int EK()
{
    int cost=0;
    while (spfa())
    {
        int t=incf[T];
        cost+=t*d[T];
        for (int i=T;i!=S;i=e[pre[i]^1])
        {
            f[pre[i]]-=t;
            f[pre[i]^1]+=t;
        }
    }
    return cost;
}
int main()
{
    scanf("%d",&n);
    S=0,T=n+1;
    memset(h,-1,sizeof h);
    
    int tot=0;
    for (int i=1;i<=n;i++)
    {
        scanf("%d",&s[i]);
        tot+=s[i];
        add(i,i<n?i+1:1,INF,1);
        add(i,i>1?i-1:n,INF,1);
    }
    tot/=n;
    for (int i=1;i<=n;i++)
        if (tot<s[i])
            add(S,i,s[i]-tot,0);
    	else if (tot>s[i])
            add(i,T,tot-S[i],0);
    printf("%d\n",EK());
    return 0;
}
```



**最小费用最大流解决二分图带权最优匹配**

实现这个算法需要用到链式前向星建图和循环队列

python可以用deque()模拟循环队列,当队列已满时,如果插入元素,会自动删除最旧元素然后将新元素加入

```python
# 创建一个最大长度为 5 的循环队列
circular_queue = deque(maxlen=5)
```

```c++
//二分图带权最优匹配问题模版,KM算法平替
//分配问题
//网络流建图:建立虚拟源点和虚拟汇点,从源点向二分图左部点集分别连一条权值为1的边,从二分图右部点集向汇点分别连一条权值为1的边
//将n件工作分配给n个人,某个人i做某件工作j的效益为cij,求最优分配方案和最差分配方案

//求最小费用最大流和最大费用最大流
//先求最小费用最大流,然后将所有边取反,再求最小费用最大流再取反即为最大费用最大流
#include<iostream>
#include<cstring>
#include<algorithm>

using namespace std;

//M=(n^2+n*2)*2+10
const int N=110,M=5210,INF=1e8;

int n,S,T;
int h[N],e[M],f[M],w[M],ne[M],idx;//f表示边的容量
int q[N],d[N],pre[N],incf[N];//incf表示点的流量
bool st[N];

void add(int a,int b,int c,int d)
{
    e[idx]=b,f[idx]=c,w[idx]=d,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,w[idx]=-d,ne[idx]=h[b],h[b]=idx++;
}

bool spfa()
{
    int hh=0,tt=1;//循环队列
    memset(d,0x3f,sizeof d);
    memset(incf,0,sizeof incf);
    q[0]=S,d[S]=0,incf[S]=INF;
    while (hh!=tt)
    {
        int t=q[hh++];
        if (hh==N) hh=0;//如果队头走到结尾了,队头回到0(循环队列性质)
        st[t]=false;//t出队了
        for (int i=h[t];~i;i=ne[i])//这里i表示边
        {
            int ver=e[i];
            if (f[i] && d[ver]>d[t]+w[i])
            {
                d[ver]=d[t]+w[i];
                pre[ver]=i;
                incf[ver]=min(f[i],incf[t]);
                if (!st[ver])//当前点不在队列中,插入队列
                {
                    q[tt++]=ver;
                    if (tt==N) tt=0;
                    st[ver]=true;
                }
            }
        }
        
    }
    return incf[T]>0;//判断汇点有没有流量
    
}
int EK()
{
    int cost=0;
    while (spfa())
    {
        int t=incf[T];
        cost+=t*d[T];//费用加上当前流量*当前到汇点路径的长度
        //从T向S走
        //pre表示前驱边,pre[i]^1即前驱边的反向边,e[pre[i]^1]即为前驱点
        for (int i=T;i!=S;i=e[pre[i]^1])//这里i表示点
        {
        	f[pre[i]]-=t;
            f[pre[i]^1]+=t;
        }
        
    }
    return cost;
}
int main()
{
    scanf("%d",&n);
    //普通点下标从1~2n
    S=0;
    T=n*2+1;
    memset(h,-1,sizeof h);
    
    for (int i=1;i<=n;i++)
    {
        add(S,i,1,0);//从源点到左部点分别连一条容量是1,费用是0的边
        add(n+i,T,1,0);//从右部点向汇点分别连一条容量是1,费用是0的边
    }
    for (int i=1;i<=n;i++)
        for (int j=1;j<=n;j++)
        {
            int c;
            scanf("%d",&c);//费用
            add(i,n+j,1,c)//连接左右点集之间的边,容量是1,费用是c
        }
    printf("%d\n",EK());
    for (int i=0;i<idx;i+=2)
    {
        f[i]+=f[i^1],f[i^1]=0;//将反向边容量加回正向边,然后将反向边容量清空
        w[i]=-w[i],w[i^1]=-w[i^1];//将每条边的费用取反
    }
    printf("%d\n",-EK());//答案记得取反
    return 0;
}
```



```c++
//最大权不相交路径问题
//数字梯形问题
//费用流+拆点+虚拟源点+虚拟汇点
#include<iostream>
#include<algorithm>
#include<cstring>

using namespace std;

const int N=1200,M=4000,INF=1e8;

int m,n,S,T;
int h[N],e[M],f[M],w[M],ne[M],idx;
int q[N],d[N],pre[N],incf[N];
bool st[N];
int id[40][40],cost[40][40];

void add(int a,int b,int c,int d)
{
    e[idx]=b,f[idx]=c,w[idx]=d,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,w[idx]=-d,ne[idx]=h[b],h[b]=idx++;
}

bool spfa()
{
    int hh=0,tt=1;
    memset(d,-0x3f,sizeof d);//求最长路,d初始成负无穷
    memset(incf,0,sizeof incf);
    q[0]=S,d[S]=0,incf[S]=INF;
    while (hh!=tt)
    {
        int t=q[hh++];
        if (hh==N) hh=0;
        st[t]=false;
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (f[i] && d[ver]<d[t]+w[i])
            {
                d[ver]=d[t]+w[i];
                pre[ver]=i;
                incf[ver]=min(f[i],incf[t]);
                if (!st[ver])
                {
                    q[tt++]=ver;
                    if (tt==N) tt=0;
                    st[ver]=true;
                }
                
            }
        }
    }
    return incf[T]>0;
}

int EK()
{
    int cost=0;
    while (spfa())
    {
        int t=incf[T];
        cost+=t*d[T];
        for (int i=T;i!=S;i=e[pre[i]^1])
        {
            f[pre[i]]-=t;
            f[pre[i]^1]+=t;
        }
    }
    return cost;
    
}
int main()
{
    int cnt=0;
    scanf("%d%d",&m,&n);
    S=++cnt;
    T=++cnt;
    for (int i=1;i<=n;i++)
        for (int j=1;j<=m+i-1;j++)
        {
            scanf("%d",&w[i][j]);
            id[i][j]=++cnt;
        }
    //规则1
    memset(h,-1,sizeof h),idx=0;
    for (int i=1;i<=n;i++)
        for (int j=1;j<=m+i-1;j++)
        {
            add(id[i][j]*2,id[i][j]*2+1,1,cost[i][j]);
            if (i==1) add(S,id[i][j]*2,1,0);
            if (i==n) add(id[i][j]*2+1,T,1,0);
            if (i<n)
            {
                add(id[i][j]*2+1,id[i+1][j]*2,1,0);
                add(id[i][j]*2+1,id[i+1][j+1]*2,1,0);
            }
        }
    printf("%d\n",EK());
    
    //规则2
    memset(h,-1,sizeof h),idx=0;
    for (int i=1;i<=n;i++)
        for (int j=1;j<=m+i-1;j++)
        {
            add(id[i][j]*2,id[i][j]*2+1,INF,cost[i][j]);
            if (i==1) add(S,id[i][j]*2,1,0);
            if (i==n) add(id[i][j]*2+1,T,INF,0);
            if (i<n)
            {
                add(id[i][j]*2+1,id[i+1][j]*2,1,0);
                add(id[i][j]*2+1,id[i+1][j+1]*2,1,0);
            }
        }
    printf("%d\n",EK());
    
    //规则3
    memset(h,-1,sizeof h),idx=0;
    for (int i=1;i<=n;i++)
        for (int j=1;j<=m+i-1;j++)
        {
            add(id[i][j]*2,id[i][j]*2+1,INF,cost[i][j]);
            if (i==1) add(S,id[i][j]*2,1,0);
            if (i==n) add(id[i][j]*2+1,T,INF,0);
            if (i<n)
            {
                add(id[i][j]*2+1,id[i+1][j]*2,INF,0);
                add(id[i][j]*2+1,id[i+1][j+1]*2,INF,0);
            }
        }
    printf("%d\n",EK());
    return 0;
}
```



**费用流之网格图模型**

```c++
//k取方格数
//拆点+最大费用最大流
#include<iostream>
#include<algorithm>
#include<cstring>

using namespace std;

const int N=5010,M=20010,INF=1e8;

int n,k,S,T;
int h[N],e[M],f[M],w[M],ne[M],idx;
int q[N],d[N],pre[N],incf[N];
bool st[N];

int get(int x,int y,int t)
{
    return (x*n+y)*2+t;
}

void add(int a,int b,int c,int d)
{
    e[idx]=b,f[idx]=c,w[idx]=d,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,w[idx]=-d,ne[idx]=h[b],h[b]=idx++;
}

bool spfa()
{
    int hh=0,tt=1;
    memset(d,-0x3f,sizeof d);
    memset(incf,0,sizeof incf);
    q[0]=S,d[S]=0,incf[S]=INF;
    while (hh!=tt)
    {
        int t=q[hh++];
        if (hh==N) hh=0;
        st[t]=false;
        
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (f[i] && d[ver]<d[t]+w[i])
            {
                d[ver]=d[t]+w[i];
                pre[ver]=i;
                incf[ver]=min(f[i],incf[t]);
                if (!st[ver])
                {
                    q[tt++]=ver;
                    if (tt==N) tt=0;
                    st[ver]=true;
                }
            }
        }
    }
    return incf[T]>0;
}
int EK()
{
    int cost=0;
    while (spfa())
    {
        int t=incf[T];
        cost+=t*d[T];
        for (int i=T;i!=S;i=e[pre[i]^1])
        {
            f[pre[i]]-=t;
            f[pre[i]^1]+=t;
        }
    }
    return cost;
}
int main()
{
   	scanf("%d%d",&n,&k);
    S=2*n*n,T=S+1;
    memset(h,-1,sizeof h);
    add(S,get(0,0,0),k,0);
    add(get(n-1,n-1,1),T,k,0);
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
        {
            int c;
            scanf("%d",&c);
            add(get(i,j,0),get(i,j,1),1,c);
            add(get(i,j,0),get(i,j,1),INF,0);
            if (i+1<n) add(get(i,j,1),get(i+1,j,0),INF,0);
            if (j+1<n) add(get(i,j,1),get(i,j+1,0),INF,0);
        }
    printf("%d\n",EK());
    return 0;
}
```



```c++
//深海机器人问题
//最大费用最大流
#include<iostream>
#include<cstring>
#include<algorithm>

using namespace std;

const int N=260,M=2000,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],w[M],ne[M],idx;
int q[N],d[N],pre[N],incf[N];
bool st[N];

int get(int x,int y)
{
    return x*(m+1)+y;
}
void add(int a,int b,int c,int d)
{
    e[idx]=b,f[idx]=c,w[idx]=d,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,w[idx]=-d,ne[idx]=h[b],h[b]=idx++;
}

bool spfa()
{
    int hh=0,tt=1;
    memset(d,-0x3f,sizeof d);
    memset(incf,0,sizeof incf);
    q[0]=S,d[S]=0,incf[S]=INF;
    while (hh!=tt)
    {
        int t=q[hh++];
        if (hh==N) hh=0;
        st[t]=false;
        
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (f[i] && d[ver]<d[t]+w[i])
            {
                d[ver]=d[t]+w[i];
                pre[ver]=i;
                incf[ver]=min(f[i],incf[t]);
                if (!st[ver])
                {
                    q[tt++]=ver;
                    if (tt==N) tt=0;
                    st[ver]=true;
                }
            }
        }
    }
    return incf[T]>0;
}
int EK()
{
    int cost=0;
    while (spfa())
    {
        int t=incf[T];
        cost+=t*d[T];
        for (int i=T;i!=S;i=e[pre[i]^1])
        {
            f[pre[i]]-=t;
            f[pre[i]^1]+=t;
        }
    }
    return cost;
}
int main()
{
    int A,B;
    scanf("%d%d%d%d",&A,&B,&n,&m);
    S=(n+1)*(m+1),T=S+1;
    memset(h,-1,sizeof h);
    for (int i=0;i<=n;i++)
        for (int j=0;j<m;j++)
        {
            int c;
            scanf("%d",&c);
            add(get(i,j),get(i,j+1),1,c);
            add(get(i,j),get(i,j+1),INF,0);
        }
    for (int i=0;i<=m;i++)
        for (int j=0;j<n;j++)
        {
            int c;
            scanf("%d",&c);
            add(get(j,i),get(j+1,i),1,c);
            add(get(j,i),get(j+1,i),INF,0);
        }
    while (A--)
    {
        int k,x,y;
        scanf("%d%d%d",&k,&x,&y);
        add(S,get(x,y),k,0);
    }
    while (B--)
    {
        int r,x,y;
        scanf("%d%d%d",&r,&x,&y);
        add(get(x,y),T,r,0);
    }
    printf("%d\n",EK());
    return 0;
}
```



```c++
//餐巾计划问题
//费用流拆点,最小费用最大流
#include<iostream>
#include<cstring>
#include<algorithm>

using namespace std;

const int N=1610,M=10000,INF=1e8;

int n,p,x,xp,y,yp,S,T;
int h[N],e[M],f[M],w[M],ne[M],idx;
int q[N],d[N],pre[N],incf[N];
bool st[N];

void add(int a,int b,int c,int d)
{
    e[idx]=b,f[idx]=c,w[idx]=d,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,w[idx]=-d,ne[idx]=h[b],h[b]=idx++;
}

bool spfa()
{
    int hh=0,tt=1;
    memset(d,0x3f,sizeof d);
    memset(incf,0,sizeof incf);
    q[0]=S,d[S]=0,incf[S]=INF;
    while (hh!=tt)
    {
        int t=q[hh++];
        if (hh==N) hh=0;
        st[t]=false;
        
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (f[i] && d[ver]<d[t]+w[i])
            {
                d[ver]=d[t]+w[i];
                pre[ver]=i;
                incf[ver]=min(f[i],incf[t]);
                if (!st[ver])
                {
                    q[tt++]=ver;
                    if (tt==N) tt=0;
                    st[ver]=true;
                }
            }
        }
    }
    return incf[T]>0;
}
int EK()
{
    int cost=0;
    while (spfa())
    {
        int t=incf[T];
        cost+=t*d[T];
        for (int i=T;i!=S;i=e[pre[i]^1])
        {
            f[pre[i]]-=t;
            f[pre[i]^1]+=t;
        }
    }
    return cost;
}
int main()
{
    scanf("%d%d%d%d%d%d",&n,&p,&x,&xp,&y,&yp);
    S=0,T=n*2+1;
    memset(h,-1,sizeof h);
    for (int i=1;i<=n;i++)
    {
        int r;
        scanf("%d",&r);
        add(S,i,r,0);
        add(S,n+i,INF,p);
        if (i<n) add(i,i+1,INF,0);
        if (i+x<=n) add(i,n+i+x,INF,xp);
        if (i+y<=n) add(i,n+i+y,INF,yp);
        
    }
    printf("%d\n",EK());
    return 0;
}
```



```c++
//志愿者招募
//无源汇有上下界最小费用可行流
#include<iostream>
#include<cstring>
#include<algorithm>

using namespace std;

const int N=1010,M=24010,INF=1e8;

int n,m,S,T;
int h[N],e[M],f[M],w[M],ne[M],idx;
int q[N],d[N],pre[N],incf[N];
bool st[N];

void add(int a,int b,int c,int d)
{
    e[idx]=b,f[idx]=c,w[idx]=d,ne[idx]=h[a],h[a]=idx++;
    e[idx]=a,f[idx]=0,w[idx]=-d,ne[idx]=h[b],h[b]=idx++;
}

bool spfa()
{
    int hh=0,tt=1;
    memset(d,0x3f,sizeof d);
    memset(incf,0,sizeof incf);
    q[0]=S,d[S]=0,incf[S]=INF;
    while (hh!=tt)
    {
        int t=q[hh++];
        if (hh==N) hh=0;
        st[t]=false;
        
        for (int i=h[t];~i;i=ne[i])
        {
            int ver=e[i];
            if (f[i] && d[ver]<d[t]+w[i])
            {
                d[ver]=d[t]+w[i];
                pre[ver]=i;
                incf[ver]=min(f[i],incf[t]);
                if (!st[ver])
                {
                    q[tt++]=ver;
                    if (tt==N) tt=0;
                    st[ver]=true;
                }
            }
        }
    }
    return incf[T]>0;
}
int EK()
{
    int cost=0;
    while (spfa())
    {
        int t=incf[T];
        cost+=t*d[T];
        for (int i=T;i!=S;i=e[pre[i]^1])
        {
            f[pre[i]]-=t;
            f[pre[i]^1]+=t;
        }
    }
    return cost;
}
int main()
{
    scanf("%d%d",&n,&m);
    S=0,T=n+2;
    memset(h,-1,sizeof h);
    int last=0;
    for (int i=1;i<=n;i++)
    {
        int cur;
        scanf("%d",&cur);
        if (last>cur) add(S,i,last-cur,0);
        else if (last<cur) add(i,T,cur-last,0);
        add(i,i+1,INF-cur,0);
        last=cur;
    }
    add(S,n+1,last,0);
    while (m--)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(b+1,a,INF,0);
    }
    printf("%d\n",EK());
    return 0;
}
```





# AcWing进阶课:欧拉路径和欧拉回路

```mysql
1.对于无向图,所有边都是连通的。
(1)存在欧拉路径的充分必要条件:度数为奇数的点只能有0或两个。(0个代表起点终点一样,2个代表起点终点不一样)
(2)存在欧拉回路的充分必要条件:度数为奇数的点只能有0个。(起点终点一样)
2.对于有向图,所有边都是连通。
(1)存在欧拉路径的充分必要条件:要么所有点的出度均等于入度;要么除了两个点之外,其余所有点的出度等于入度,剩余的两个点:
一个满足出度比入度多1(起点),另一个满足入度比出度多1(终点)
(2)存在欧拉回路的充分必要条件:所有点的出度均等于入度
```

# AcWing进阶课:基环树dp

![avatar](/Users/kaosdragon/Desktop/算法/pictures/基环树.png)

```mysql
基环树是特殊的仙人掌,只有一个环的仙人掌就是基环树
基环树:n个点n条边的连通无向图,即在树上加一条边构成的恰好包含一个环的图,如果不保证连通,n个点n条边的无向图也可能是若干棵基环树组成       的森林,简称为基环树森林
外向树:对于有向图,环上边的方向一致,树上方向从父节点指向子节点
内向树:对于有向图,环上边的方向一致,树上方向从子节点指向父节点
多棵外向树构成外向树森林,多棵内向树构成内向树森林
基环树特性:一个环上挂着若干棵树,n个点,n条边
注意一个图如果有n个点和n条边,不一定是基环树,比如这个图可能不连通
判断基环树:对于有向图,每个点恰好有一个出边或者一个入边,
		 无向图:从每个点出发有一条唯一的边
		 有向图:每个点有且仅有一条出边(内向树)或者每个点有且仅有一条入边(外向树)

```



![avatar](/Users/kaosdragon/Desktop/算法/pictures/基环树2.png)

```mysql
求基环树两点间最大距离:
分两种情况讨论:
1.两个点在一棵树上:转化成树型dp
2.两个点不在一棵树上,那么就必然要经过环,距离可以分解成d(x)+d(x,y)+d(y),d(x),d(y)表示两段树上的距离,d(x,y)表示环上距离s
  s可以用前缀和加工变成sx-sy,表达式就可以转化成dx+sx-sy+dy即dx+sx+dy-sy,当我们固定x时,dx+sx固定,只需要枚举y
  此时相当于求环上两点间最大距离问题,我们扩环成链,将环拉伸成长度为n的链,再延长一倍,然后在这个长度为2n的链上,求长度为n的滑动窗口   的最大值,而滑动窗口的最值可以用单调队列解决。同时注意dx,dy这种从环上某一点延伸出来的距离可以预处理算出来
```

# AcWing进阶课:模拟退火

启发式搜索:如遗传算法,模拟退火,爬山法等

随机算法寻找最优解

```mysql
1.温度:步长,
初始温度:根据数据范围来定,如果数据范围10^5,初始温度就可以取一半,5x10^4
终止温度:10^-5
衰减系数:指数衰减,衰减系数越接近于1,衰减越慢,找到最优解的概率越大
温度不断降低
2.随机选择一个点,f(新点)-f(旧点)=delta
情况1:delta<0,则跳到新点
情况2:delta>0,则以一定概率跳过去,p=e^(-delta/T)
模拟退火可能找到的是局部最优解而不是全局最优解,因此我们需要迭代若干次,如迭代100次

随机初始点,定义初始温度,定义精度(终止温度),衰减系数
模拟退火必须保证函数具有连续性
```



```mysql
技巧
卡时
while(time<0.8s)
```



```c++
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <ctime>
using namespace std;

#define x first
#define y second

typedef pair<double,double>PDD;
const int N=110;

int n;
PDD q[N];
double ans=1e8;//全局最优解

double rand(double l,double r)
{
    return (double)rand()/RAND_MAX*(r-l)+l;
}

//求两点间距离,结合calc使用,calc根据不同题的情况进行修改
double get_dist(PDD a,PDD b)
{
    double dx=a.x-b.x;
    double dy=a.y-b.y;
    return sqrt(dx*dx+dy*dy);
}
double calc(PDD p)
{
    double res=0;
    for (int i=0;i<n;i++)
        res+=get_dist(p,q[i]);
    ans=min(ans,res);
	return res;
}
void simulate_anneal()
{
    PDD cur(rand(0,10000),rand(0,10000));
    for (double t=1e4;t>1e-4;t*=0.99)
    {
        PDD np(rand(cur.x-t,cur.x+t),rand(cur.y-t,cur.y+t));
        double dt=calc(np)-calc(cur);
        if (exp(-dt/t)>rand(0,1)) cur=np;
        
    }
}
int main()
{
    scanf("%d",&n);
    for (int i=0;i<n;i++) scanf("%lf%lf",&q[i].x,&q[i].y);
    //迭代100次
    for (int i=0;i<100;i++) simulate_anneal();
    printf("%.0lf\n",ans);
    return 0;
}
```

# AcWing进阶课:爬山法

牛顿迭代法也可以类似地求极值

$X_{n+1}=X_n-\frac{f(X_n)}{f(X_n')}$

牛顿迭代法是爬山法的一种

```c++
//爬山法要求函数必须是凸函数
#include<iostream>
#include<cstring>
#include<algorithm>
#include<cmath>

using namespace std;

const int N=15;

int n;
double d[N][N];
double ans[N],dist[N],delta[N];

void calc()
{
    double avg=0;
    for (int i=0;i<n;i++)
    {
        dist[i]=delta[i]=0;
        for (int j=0;j<n;j++)
            dist[i]+=(d[i][j]-ans[j])*(d[i][j]-ans[j]);
        dist[i]=sqrt(dist[i]);
        avg+=dist[i]/(n+1);
    }
    for (int i=0;i<n+1;i++)
        for (int j=0;j<n;j++)
            delta[j]+=(dist[i]-avg)*(d[i][j]-ans[j])/avg;
    
        
}
int main()
{
    scanf("%d",&n);
    for (int i=0;i<n+1;i++)
        for (int j=0;j<n;j++)
        {
            scanf("%lf",&d[i][j]);
            ans[j]+=d[i][j]/(n+1);
            
        }
    for (double t=1e4;t>1e-6;t*=0.99997)
    {
        calc();
        for (int i=0;i<n;i++)
            ans+=delta[i]*t;
    }
    for (int i=0;i<n;i++) printf("%.3lf",ans[i]);
    return 0;
}
```

# AcWing进阶课:后缀数组

后缀数组介绍:

三种写法:

1.倍增,O(nlogn)

2.DC3,O(n),常数大,代码难写

3.SAIS



该实现下标从1开始

如果某个下标是i,那么从该下标开始的后缀被称为第i个后缀

倍增:n个后缀,O(nlogn)时间内将所有后缀按字典序排序



几个参数:

**sa数组**:sa[i]表示排名第i位的是第几个后缀

**rk数组**:rk[i]表示第i个后缀的排名是多少

sa和rk互逆

**height数组**:sa[i]与sa[i-1]的最长公共前缀(LCP)



倍增+基数排序

第一次只看每个后缀的第一个字符,如果关键字相同,按照排序前的相对顺序

基数排序对后缀串排序

将长度为k的关键字离散化,双关键字排序



基数排序,统计1～n(数值)范围内每种数字出现的个数,然后对个数计算前缀和

为了考虑相同元素,前缀和每次用完要减1

每个数字对应的前缀和是多少,排名就是多少

为了保证排序的稳定性,要从后往前枚举



```c++
//倍增求后缀数组代码
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=1000010;

int n,m;
char s[N];
int sa[N],x[N],y[N],c[N],rk[N],height[N];

void get_sa()
{
    //按照第一个关键字排序
    for (int i=1;i<=n;i++) c[x[i]=s[i]]++; //离散化
    for (int i=2;i<=m;i++) c[i]+=c[i-1]; //前缀和
    for (int i=n;i;i--) sa[c[x[i]]--]=i; //从后往前确定每个数的排名
    
    for (int k=1;k<=n;k<<=1)
    {
        int num=0;
        //将所有后缀按照第二个关键字排序
        //没有第二个关键字用空字符补齐
        for (int i=n-k+1;i<=n;i++) y[++num]=i;
        for (int i=1;i<=n;i++)
            if (sa[i]>k)
                y[++num]=sa[i]-k;
        //按照第一个关键字排序
        for (int i=1;i<=m;i++) c[i]=0;
        for (int i=1;i<=n;i++) c[x[i]]++;
        for (int i=2;i<=m;i++) c[i]+=c[i-1];
        for (int i=n;i;i--) sa[c[x[y[i]]]--]=y[i],y[i]=0;
        //将x信息存到y
        swap(x,y);
        //离散化
        x[sa[1]]=1,num=1;
        for (int i=2;i<=n;i++)
            x[sa[i]]=(y[sa[i]]==y[sa[i-1]]&&y[sa[i]+k]==y[sa[i-1]+k])?num:++num;
        if (num==n) break;
        m=num;     
    }
}
void get_height()
{
    for (int i=1;i<=n;i++) rk[sa[i]]=i; //rk表示第i个后缀的排名
    for (int i=1,k=0;i<=n;i++)
    {
        if (rk[i]==1) continue;
        if (k) k--;
        int j=sa[rk[k]-1];
        while (i+k<=n && j+k<=n && s[i+k]==s[j+k]) k++;
        height[rk[i]]=k;
    }
}

int main()
{
    scanf("%s",s+1);
    n=strlen(s+1),m=122;
    get_sa();
    get_height();
    for (int i=1;i<=n;i++) printf("%d ",sa[i]);
    puts("");
    for (int i=1;i<=n;i++) printf("%d ",height[i]);
    puts("");
    return 0;
}
```



```python
#python实现倍增+基数排序求后缀数组
class SuffixArray:
    def __init__(self, s):
        self.s = [0] + list(map(ord, s))
        self.n = len(s)
        self.m = 122
        self.N = max(self.n + 5, self.m + 5)
        self.x = [0] * self.N
        self.y = [0] * self.N
        self.c = [0] * self.N
        self.sa = [0] * self.N
        self.rk = [0] * self.N
        self.height = [0] * self.N

    # 下标从1开始
    def get_sa(self):
        for i in range(1, self.n + 1):
            self.x[i] = self.s[i];
            self.c[self.x[i]] += 1
        for i in range(2, self.m + 1):
            self.c[i] += self.c[i - 1]
        for i in range(self.n, 0, -1):
            self.sa[self.c[self.x[i]]] = i;
            self.c[self.x[i]] -= 1

        k = 1
        while k <= self.n:
            num = 0
            for i in range(self.n - k + 1, self.n + 1):
                num += 1;
                self.y[num] = i
            for i in range(1, self.n + 1):
                if self.sa[i] > k:
                    num += 1;
                    self.y[num] = self.sa[i] - k
            for i in range(1, self.m + 1):
                self.c[i] = 0
            for i in range(1, self.n + 1): self.c[self.x[i]] += 1
            for i in range(2, self.m + 1): self.c[i] += self.c[i - 1]
            for i in range(self.n, 0, -1): self.sa[self.c[self.x[self.y[i]]]] = self.y[i];self.c[
                self.x[self.y[i]]] -= 1;self.y[i] = 0
            self.x, self.y = self.y, self.x
            self.x[self.sa[1]] = 1;
            num = 1

            for i in range(2, self.n + 1):
                if self.y[self.sa[i]] == self.y[self.sa[i - 1]] and self.y[self.sa[i] + k] == self.y[
                    self.sa[i - 1] + k]:
                    self.x[self.sa[i]] = num
                else:
                    num += 1
                    self.x[self.sa[i]] = num

            if num == self.n: break
            self.m = num

            k <<= 1
        return self.sa

    def get_height(self):
        for i in range(1, self.n + 1):
            self.rk[self.sa[i]] = i
        k = 0
        for i in range(1, self.n + 1):
            if self.rk[i] == 1:
                continue
            if k: k -= 1
            j = self.sa[self.rk[i] - 1]
            while i + k <= self.n and j + k <= self.n and self.s[i + k] == self.s[j + k]:
                k += 1
            self.height[self.rk[i]] = k
        return self.height
```



## 极快的SA-IS实现

```python
def inducedSort(s: List[int], sa1: List[int], t: List[int], m=26):
    n = len(s)
    cnt = [0] * m
    for i in range(n):
        cnt[s[i]] += 1
    for i in range(1, m):
        cnt[i] += cnt[i - 1]
    start, end = [0] + cnt[:-1], cnt[:]
    sa = [-1] * n + [n]
    for x in reversed(sa1[1:]):
        cnt[s[x]] -= 1
        sa[cnt[s[x]]] = x
    for i in range(-1, n):
        if sa[i] > 0:
            c = sa[i] - 1
            if not t[c]:
                sa[start[s[c]]] = c
                start[s[c]] += 1
    for i in range(n - 1, -1, -1):
        if sa[i] > 0:
            c = sa[i] - 1
            if t[c]:
                end[s[c]] -= 1
                sa[end[s[c]]] = c
    return sa[:-1]

def SA_IS(s: List[int], m=26):
    n = len(s)
    t = [False] * (n + 1)
    for i in reversed(range(n - 1)):
        t[i] = t[i + 1] if s[i] == s[i + 1] else (s[i] < s[i + 1])
    critical = list()
    for i in range(1, n):
        if t[i] and not t[i - 1]:
            critical.append(i)
    nc = len(critical)
    index = [-1] * n + [n]
    for i, x in enumerate(critical):
        index[x] = i
    sa0 = inducedSort(s, [n] + critical, t, m)
    s1 = [0] * (nc + 1)
    critical.append(n)
    last, p, repeat = "", 0, False
    for x in sa0:
        if index[x] >= 0:
            c = s[x : critical[index[x] + 1]]
            if c != last:
                p += 1
                last = c
            else:
                repeat = True
            s1[index[x]] = p
    if repeat:
        sa1 = [critical[x] for x in SA_IS(s1, p + 1)]
    else:
        sa1 = [n] + [x for x in sa0 if index[x] >= 0]
    return inducedSort(s, sa1, t, m)

def suffixArray(s: str) -> (List[int], List[int], List[int]):
    n, k = len(s), 0
    sa = SA_IS([ord(x) - 97 for x in s])
    rk = [0] * n
    for i in range(n):
        rk[sa[i]] = i
    height = [0] * n
    s += '#'
    for i in range(n):
        if rk[i]:
            if k > 0:
                k -= 1
            while s[i + k] == s[sa[rk[i] - 1] + k]:
                k += 1
            height[rk[i]] = k
    return rk, sa, height

```



下面我详细解释一下这段代码。整体来说，这段代码实现了构造后缀数组（suffix array）的算法，其中采用了 **SA-IS** 算法。SA-IS 算法是一种线性时间构造后缀数组的方法，其核心思想是利用“归纳排序”（induced sorting）的方式对后缀进行排序，并利用递归将问题规模缩小。代码中还计算了辅助数组——**rank 数组**（后缀的逆序，表示每个位置对应的后缀在后缀数组中的排名）以及 **height 数组**（也称为 LCP 数组，用来存储相邻后缀的最长公共前缀长度）。

下面逐个函数来说明代码的实现细节：

------

### 1. 函数 `inducedSort(s: List[int], sa1: List[int], t: List[int], m=26)`

**功能：**
 利用已知的 LMS（Left-most S-type）后缀的排序结果，对整个后缀数组进行“归纳排序”。
 **参数说明：**

-   `s`: 整数列表，表示原字符串，其中每个字符已转换为一个整数（代码中假设字母是小写，所以用 `ord(x) - 97`）。
-   `sa1`: 已排序的 LMS 后缀下标列表。注意第一个元素是 `n`（哨兵/末尾标记），后面跟着 LMS 后缀的下标。
-   `t`: 布尔列表，用来标记每个位置的后缀类型。通常，`t[i] = True` 表示该位置是 S 型（后缀字典序较小），`False` 表示 L 型（后缀字典序较大）。
-   `m`: 字母表的大小，默认 26（适用于英文小写字母）。

**主要步骤：**

1.  **统计每个字符的出现次数：**
     使用 `cnt` 数组统计字符串中每个字符（整数）出现的次数。
2.  **确定桶的边界：**
     对 `cnt` 数组做前缀和处理，得到每个字符在排序数组中的“结束位置”（即桶的右边界）；同时构造 `start` 数组表示每个桶的起始位置。
3.  **初始化后缀数组：**
     创建一个大小为 `n+1` 的数组 `sa`，初始值为 -1（代表未填充），并在最后加上一个哨兵 `n`。
4.  **放置 LMS 后缀：**
     根据 `sa1` 中 LMS 下标的排序，从后向前扫描，将 LMS 后缀放入对应的桶中（根据最后一个字符决定其桶的位置），每放入一个后缀，更新桶尾指针（`cnt[s[x]]`）。
5.  **归纳排序 L 型后缀：**
     扫描已经部分构造好的 `sa`，对于每个有效位置（`sa[i] > 0`），检查它前一个位置 `c = sa[i] - 1`。如果 `c` 属于 L 型（即 `t[c]` 为 False），则将其放到对应桶的起始位置，并更新桶头指针（`start[s[c]]`）。
6.  **归纳排序 S 型后缀：**
     反向扫描 `sa`，对每个位置 `sa[i]`，检查其前一位置 `c = sa[i] - 1`。如果 `c` 是 S 型（即 `t[c]` 为 True），则将其放入对应桶的末尾（即 `end[s[c]]`），并更新桶尾指针。

最后返回排序好的后缀数组（去掉最后那个哨兵）。

------

### 2. 函数 `SA_IS(s: List[int], m=26)`

**功能：**
 利用 SA-IS 算法构造后缀数组。整个算法分为以下几个关键步骤：

1.  **标记后缀类型：**
    -   构造数组 `t`，用来标记每个位置的后缀类型。
    -   从后往前判断：若相邻字符相等，则与后面的类型一致；否则，若 `s[i] < s[i+1]`，则 `t[i]` 为 S 型（True），否则为 L 型（False）。
2.  **确定 LMS 位置：**
    -   LMS（Left-most S-type）位置定义为那些 S 型但其前一个位置为 L 型的位置。
    -   将所有 LMS 下标保存在 `critical` 列表中。
3.  **映射 LMS 下标：**
    -   创建 `index` 数组，将 LMS 下标在原字符串中的位置映射到 `critical` 列表中的序号，用于后续区分不同的 LMS 子串。
4.  **初步归纳排序：**
    -   调用 `inducedSort` 函数，利用 `[n] + critical` 作为初始的 LMS 排序输入，得到一个初步排序的后缀数组 `sa0`。
5.  **构造简化问题（Reduced String）：**
    -   利用 `sa0` 中 LMS 后缀的顺序，对 LMS 子串进行命名（即比较每个 LMS 子串与上一个 LMS 子串是否相同，若不同则赋予新的编号）。
    -   构造一个新的字符串 `s1`（实际上是 LMS 子串对应的编号序列），该字符串的长度为 LMS 个数加 1（末尾的哨兵）。
6.  **递归求解或直接确定 LMS 排序：**
    -   如果在构造 `s1` 的过程中出现重复（即某些 LMS 子串相同），则对 `s1` 递归调用 SA-IS 算法以确定 LMS 子串的正确顺序；
    -   否则，可以直接利用 `s1` 得到 LMS 后缀在原字符串中的顺序。
7.  **最后归纳排序：**
    -   利用得到的 LMS 后缀的正确顺序，再次调用 `inducedSort`，完成整个后缀数组的构造。

最终返回构造好的后缀数组。

------

### 3. 函数 `suffixArray(s: str) -> (List[int], List[int], List[int])`

**功能：**
 该函数是整个实现的入口，用于处理输入字符串并最终返回三个数组：

-   **rk（rank 数组）**：后缀在后缀数组中的排名，即 `rk[i]` 表示从位置 `i` 开始的后缀在排序后的后缀数组中的位置。
-   **sa（后缀数组）**：存储后缀起始下标的数组，按字典序排序。
-   **height（LCP 数组）**：表示相邻后缀之间的最长公共前缀长度（Longest Common Prefix）。

**主要步骤：**

1.  **转换字符为整数：**
    -   将输入字符串 `s` 中的每个字符转换为对应的整数（通过 `ord(x) - 97`），得到整数列表传入 SA-IS 算法。
    -   注意：这种转换假设输入只包含小写字母。
2.  **调用 SA-IS 构造后缀数组：**
    -   得到后缀数组 `sa` 后，再构造 **rank 数组**。
    -   通过遍历 `sa`，令 `rk[sa[i]] = i`。
3.  **计算 LCP 数组（height 数组）：**
    -   对于每个位置 `i`，如果 `rk[i]` 非零（即不是后缀数组中排在第一的后缀），则：
        -   利用前一次计算的 LCP 值 `k`（如果 `k>0` 则先减 1），比较从位置 `i+k` 和 `sa[rk[i]-1]+k` 开始的字符，直到不相等为止，记录该公共前缀的长度到 `height[rk[i]]`。
    -   这里需要注意，为了防止数组越界，代码中在字符串末尾附加了一个特殊字符 `'#'`。

最后，函数返回 `rk`、`sa` 和 `height` 三个数组。

------

### 总结

-   **SA-IS 算法核心思想：**
    1.  利用字符比较标记出每个后缀是 S 型还是 L 型。
    2.  找出 LMS 后缀（即 S 型中第一个出现的位置），并初步排序。
    3.  将 LMS 后缀构造成一个简化问题（缩小规模），递归求解后 LMS 后缀的顺序。
    4.  利用 LMS 后缀的顺序，通过归纳排序（induced sorting）将其他后缀排序，最终得到整个后缀数组。
-   **辅助数组：**
    -   **rank 数组**：记录每个后缀在后缀数组中的位置。
    -   **height 数组（LCP 数组）**：记录相邻后缀之间的最长公共前缀长度，对许多字符串处理问题（如查找重复子串）非常有用。

这段代码结构清晰，通过分步实现 SA-IS 算法的核心过程，从而高效地构造后缀数组及其辅助信息。



# AcWing进阶课:后缀自动机



# AcWing进阶课:斜率优化dp



# AcWing进阶课:CDQ分治

中文名陈丹琦分治

本质求解三维偏序问题

对于每个i,能够快速找出来有多少个j,使得$a_j\leq a_i,b_j\leq b_i,c_j\leq c_i(i\neq j)$

```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N=100010,M=200010;

int n,m;
struct Data
{
    int a,b,c,s,res;
    bool operator< (const Data& t) const
    {
        if (a!=t.a) return a<t.a;
        if (b!=t.b) return b<t.b;
        return c<t.c;
    }
    bool operator==(const Data& t) const
    {
        return a==t.a && b==t.b && c==t.c;
    }
}q[N],w[N];
int tr[M],ans[N];

int lowbit(int x)
{
    return x&-x;
}
void add(int x,int v)
{
    for (int i=x;i<M;i+=lowbit(i)) tr[i]+=v;
}
int query(int x)
{
    int res=0;
    for (int i=x;i;i-=lowbit(i)) res+=tr[i];
    return res;
}
void merge_sort(int l,int r)
{
    if (l>=r) return;
    int mid=l+r>>1;
    merge_sort(l,mid),merge_sort(mid+1,r);
    int i=l,j=mid+1,k=0;
    while (i<=mid && j<=r)
        if (q[i].b<=q[j].b) add(q[i].c,q[i].s,w[k++]=q[i++]);
    	else q[j].res+=query(q[j].c),w[k++]=q[j++];
    while (i<=mid) add(q[i].c,q[i].s),w[k++]=q[i++];
    while (j<=r) q[j].res=query(q[j].c),w[k++]=q[j++];
    for (i=l;i<=mid;i++) add(q[i].c,-q[i].s);
    for (i=l,j=0;j<k;i++,j++) q[i]=w[j];
    
}
int main()
{
    scanf("%d%d",&n,&m);
    for (int i=0;i<n;i++)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        q[i]={a,b,c,1};
    }
    sort(q,q+n);
    int k=1;
    for (int i=1;i<n;i++)
        if (q[i]==q[k-1]) q[k-1].s++;
    	else q[k++]=q[i];
    merge_sort(0,k-1);
    for (int i=0;i<k;i++)
        ans[q[i].res+q[i].s-1]+=q[i].s;
    for (int i=0;i<n;i++) printf("%d\n",ans[i]);
    return 0;
}
```



```python
#python版本
import sys
sys.setrecursionlimit(300000)

def lowbit(x):
    return x & -x

def add(x, v, BIT):
    while x < len(BIT):
        BIT[x] += v
        x += lowbit(x)

def query(x, BIT):
    res = 0
    while x:
        res += BIT[x]
        x -= lowbit(x)
    return res

class Data:
    __slots__ = ('a', 'b', 'c', 's', 'res')
    def __init__(self, a, b, c, s=1, res=0):
        self.a = a
        self.b = b
        self.c = c
        self.s = s  # 重复出现的次数
        self.res = res  # 记录比当前点小的点个数
    def __lt__(self, other):
        if self.a != other.a:
            return self.a < other.a
        if self.b != other.b:
            return self.b < other.b
        return self.c < other.c
    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.c == other.c

def merge_sort(q, l, r, BIT):
    if l >= r:
        return
    mid = (l + r) // 2
    merge_sort(q, l, mid, BIT)
    merge_sort(q, mid + 1, r, BIT)
    temp = []
    i = l
    j = mid + 1
    while i <= mid and j <= r:
        if q[i].b <= q[j].b:
            add(q[i].c, q[i].s, BIT)
            temp.append(q[i])
            i += 1
        else:
            q[j].res += query(q[j].c, BIT)
            temp.append(q[j])
            j += 1
    while i <= mid:
        add(q[i].c, q[i].s, BIT)
        temp.append(q[i])
        i += 1
    while j <= r:
        q[j].res += query(q[j].c, BIT)
        temp.append(q[j])
        j += 1
    # 回滚左半部分的树状数组更新
    for k in range(l, mid + 1):
        add(q[k].c, -q[k].s, BIT)
    q[l:r + 1] = temp

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    q_list = []
    for _ in range(n):
        a = int(next(it))
        b = int(next(it))
        c = int(next(it))
        q_list.append(Data(a, b, c))
    # 根据 a, b, c 三个维度排序
    q_list.sort()
    # 合并重复的点
    new_q = []
    for d in q_list:
        if new_q and d == new_q[-1]:
            new_q[-1].s += 1
        else:
            new_q.append(d)
    q_list = new_q
    k = len(q_list)
    # 构造树状数组 BIT，下标从 1 开始（假设 c 的取值范围为 [1, m]）
    BIT = [0] * (m + 1)
    merge_sort(q_list, 0, k - 1, BIT)
    ans = [0] * n
    # ans[i] 表示有 i 个点小于等于当前点的个数
    for d in q_list:
        ans[d.res + d.s - 1] += d.s
    sys.stdout.write("\n".join(map(str, ans)))
```

# AcWing进阶课:树链剖分

```c++
通过给树中每个节点重新编号,使得树中任意一条路径转化成O(logn)段连续区间,从而将树的操作转化成维护区间的操作,之后可以用线段树维护
1.将一棵树转化成一个序列.
2.树中路径转化成logn段连续区间

基本概念
重儿子:对于某个节点a,在其所有子树中,如果某棵子树的节点是最多的,称其为重儿子,如果有多个子树节点数量相同,则任选其一作为重儿子,其余节点称为轻儿子
轻儿子:见上述定义
重边:重儿子和其父节点之间的边称为重边,其余的边称为轻边
轻边:见上述定义
重链:极大的由重边构成的路径
    
定理:树中任意一条路径均可拆分成O(logn)个连续区间
按照dfs序变成序列,优先遍历重儿子,这样可以使得一条重链编号连续

```



# 计算几何

## 1.前置知识点

1.pi=acos(-1)

2.余弦定理:$c^2=a^2+b^2-2abcos(t)$

## 2.浮点数的比较

```c++
const double eps=1e-8;
//符号函数
int sign(double x){
    if (fabs(x)<eps) return 0;
    if (x<0) return -1;
    return 1;
}
//比较函数
int cmp(double x,double y){
    if (fabs(x-y)<eps) return 0;
    if (x<y) return -1;
    return 1;
}
```

## 3.向量

### 1.向量加减法和数乘法

(x1,y1)+(x2,y2)=(x1+x2,y1+y2)

(x1,y1)-(x2,y2)=(x1-x2,y1-y2)

$b\cdot(x,y)=(bx,by)$

### 2.向量内积

$A\cdot B=|A||B|cos(c)$

$(x1,y1)\cdot(x2,y2)=x1y1+x2y2$

代码实现

```c++
double dot(Point a,Point b){
    return a.x*b.x+a.y*b.y;
}
```

### 3.外积(叉积)

$A\times B=|A||B|sin\theta$

$(x1,y1)\times(x2,y2)=x1y2-x2y1$

注意角度表示从向量B逆时针转到向量A的角度

代码实现

```c++
double cross(Point a,Point b){
    return a.x*b.y-b.x*a.y;
}
```

### 4.常用函数

1.取模

$|A|=sqrt(A\times A)$

```c++
double get_length(Point a){
    return sqrt(dot(a,a));
}
```

2.求向量夹角

```c++
double get_angle(Point a,Point b){
    return acos(dot(a,b)/get_length(a)/get_length(b))
}
```

3.计算两个向量构成的平行四边形有向面积

```c++
double area(Point a,Point b,Point c){
    return cross(b-a,c-a);
}
```

4.将向量逆时针旋转$\theta角$
$$
(x',y')=(x,y)\cdot\begin{pmatrix}
cos\theta & sin\theta\\
-sin\theta & cos\theta
\end{pmatrix}=(xcos\theta-ysin\theta,xsin\theta+ycos\theta)
$$

```c++
double rotate(Point a,double angle){
    return Point(a.x*cos(angle)+a.y*sin(angle),-a.x*sin(angle)+a.y*cos(angle));
}
```

## 4.点与线

### 1.直线定理

1.一般式 ax+by+c=0

2.点向式 $p_0+tv,p_0,p_0代表起点,t代表系数,v代表向量$

3.斜阶式 y=kx+b

### 2.常用操作

1.判断点c在直线上

$A\times B=0$ 从直线上取两点a,b组成向量成为A,a和c构成向量为B,若A,B叉乘为0,则点在直线上

2.两直线相交,求交点

```c++
//cross(v,w)==0,首先确保两直线有交点,如果叉乘为0说明无交点
//下面公式来源于相似三角形
Point get_line_intersection(Point p,Vector v,Point q,vetor w){
    vector u=p-q;
    double t=cross(w,u)/cross(v,w);
    return p+v*t;
}
```

3.点到直线距离

```c++
double distance_to_line(Point p,Point a,Point b){
    vector v1=b-a,v2=p-a;
    return fabs(cross(v1,v2)/get_length(v1));
}
```

4.点到线段的距离

```c++
double distance_to_segment(Point p,Point a,Point b){
    if (a==b) return get_length(p-a);
    Vector v1=b-a,v2=p-a,v3=p-b;
    if (sign(dot(v1,v2))<0) return get_length(v2);
    if (sign(dot(v1,v3))>0) return get_length(v3);
    return distance_to_line(p,a,b);
}
```

5.点在直线上的投影

```c++
double get_line_projection(Point p,Point a,Point b){
    Vector v=b-a;
    return a+v*(dot(v,p-a)/dot(v,v));
}
```

6.点是否在线段上

```c++
bool on_segment(Point p,Point a,Point b){
    return sign(cross(p-a,p-b))==0 && sign(dot(p-a,p-b))<=0;
}
```

7.判断两线段是否相交(跨立实验)

```c++
bool segment_intersection(Point a1,Point a2,Point b1,Point b2){
    double c1=cross(a2-a1,b1-a1),c2=cross(a2-a1,b2-a1);
    double c3=cross(b2-b1,a2-b1),c4=(b2-b1,a1-b1);
    return sign(c1)*sign(c2)<=0 && sign(c3)*sign(c4)<=0;
}
```

## 5.多边形

### 1.三角形

1.面积

(1)叉积

(2)海伦公式

```c++
p=(a+b+c)/2;
S=sqrt(p*(p-a)*(p-b)*(p-c));
```

2.三角形四心

(1)外心,外接圆圆心

三边中垂线交点,到三角形三个顶点的距离相等

(2)内心,内切圆圆心

角平分线交点,到三边距离相等

(3)垂心

三条垂线交点

(4)重心

三条中线交点(到三角形三顶点距离的平方和最小的点,三角形内到三边距离之积最大的点)

### 2.普通多边形

通常按逆时针存储所有点

1.定义

(1)多边形

由在同一平面且不在同一直线上的多条线段首尾顺次连接且不相交所组成的图形叫多边形.

(2)简单多边形

简单多边形是除相邻边外其他边不相交的多边形

(3)凸多边形

过多边形的任意一边做一条直线,如果其他各个顶点都在这条直线的同侧,则把这个多边形叫做凸多边形.任意凸多边形外交和均为360度.任意凸多边形内角和为(n-2)*180度

2.常用函数

(1)求多边形面积(不一定是凸多边形)

我们可以从第一个顶点出发把凸多边形分成n-2个三角形,然后把面积加起来

```c++
double polygon_area(Point p[],int n){
    double s=0;
    for (int i=1;i+1<n;i++){
        s+=cross(p[i]-p[0],p[i+1]-p[i]);
    }
    return s/2;
}
```

(2)判断点是否在多边形内(不一定是凸多边形)

a.射线法.从该点任意做一条和所有边都不平行的射线.交点个数为偶数,则在多边形外,为奇数,则在多边形内.

b.转角法,在内转360度,在外转0度

(3)判断点是否在凸多边形内

只需判断点是否在所有边的左边(逆时针存储多边形)

3.皮克定理

一个计算点阵中顶点在格点(横纵坐标都是整数的点)上的多边形面积公式为:

S=a+b/2-1

其中a表示多边形内部的点数,b表示多边形边界上的点数,S表示多边形的面积.

# 凸包

```c++

```

# WQS二分

```c++
```


# 1.前缀和优化dp

**1997.访问完所有房间的第一天**

这是前缀和优化dp的经典题目

分析题意可以发现如果想到达i+1房间,就必须经过i房间偶数次
当在奇数次来到房间i后,在往前不断回访之前的房间后,会再回到房间i,然后前往i+1房间
注意回访的次数都是偶数次
设当前下标为i,nextVisit[i]=x,f[i]表示访问房间i且次数到访问房间i且次数为偶数这个周期(闭区间)需要的天数
可以发现$f[i]=s[i]-s[x](即f[x]+f[x+1]+...+f[i-1]),s[t+1]表示f[0]~f[t]的前缀和$
因此一边更新f,一边更新f的前缀和,就可以得到答案
最终答案就是访问0~n-2房间的总天数+1,因为最后一个房间只需访问一次
由于天数下标从0开始,因此最终答案为$s[n-1]+1-1=s[n-1]$

进一步优化,式子变形
$f[i]=s[i]-s[x]+2,s[i+1]=s[i]+f[i]\rightarrow s[i+1]=2*s[i]-s[x]+2$
由于最后答案为s[n-1],因此只需更新s即可
由于答案只用到s[n-1],因此s数组开n长度即可

```python
#1997(https://leetcode.cn/problems/first-day-where-you-have-been-in-all-the-rooms/)
mod=1_000_000_007
class Solution:
    def firstDayBeenInAllRooms(self, nextVisit: List[int]) -> int:
        n=len(nextVisit)
        s=[0]*n #s[i+1]表示f[0]+...+f[i]的前缀和
        for i,x in enumerate(nextVisit[:-1]):
            s[i+1]=(2*s[i]-s[x]+2)%mod
        return s[-1]
```

**3333.找到初始输入字符串II**

容斥原理+乘法原理+前缀和优化多重背包求方案数
将连续的字符看成一个物品取有限次,容易看出这是一个多重背包求方案数模型
但是直接写多重背包是三重循环,O(n^3)复杂度,因为本题物品种类数,每个物品的数量上限,背包容量,都是n
所以需要进行优化
优化1:
容斥原理:长度>=k的方案数=总方案数减去长度 <=k-1的方案数,这样就将背包容量限制在k-1

时间复杂度变成了$O(n^2k)$
优化2:
由于每种物品至少选一个,因此如果n>=k,即物品种类数>=k,每种物品随便选都可以满足条件
答案就是总方案数,直接用乘法原理算出即可
如果n<k,就用多重背包解决,此时物品数量上限变成了k-1
时间复杂度变成了$O(nk^2)$
优化3:
前缀和优化多重背包求方案数
多重背包求方案数可以用前缀和优化掉一重循环,因此最终时间复杂度为$O(k^2)$
**注意如果是求最值的完全背包问题,可以用滑动窗口/单调队列优化成二重循环**

实际实现时,由于每种物品必须至少选一个,因此提前取出来,方便后续计算

这样后面每种物品取的数量可以为0,方便优化

```python
fmax=lambda x,y:x if x>y else y
mod=1_000_000_007
class Solution:
    def possibleStringCount(self, word: str, k: int) -> int:
        #word长度<k,不合法,返回0
        m=len(word)
        if m<k:
            return 0
        arr=[]
        tot=1
        #按照连续串分段
        for x,y in groupby(word):
            size=len(list(y))
            if size>1:
                if k>0:
                    arr.append(size-1)
                #tot统计没有长度限制,每段至少选一个字符的总方案数,乘法原理
                tot=tot*size%mod
            k-=1
        #可行性剪枝:如果k<=0,说明分段数量>=k,那么每组至少选一个就能满足长度>=k的条件,直接返回tot
        if k<=0:
            return tot
        #前缀和优化多重背包求方案数
        f=[1]*k
        for x in arr:
            #计算0~k-1范围前缀和
            for j in range(1,k):
                f[j]=(f[j]+f[j-1])%mod
            #类似01背包倒序枚举
            for j in range(k-1,x,-1):
                #f[j]对应的是上一行j-x~k-1的子段和
                #用前缀和更新子数组和
                f[j]=(f[j]-f[j-x-1])%mod
        return (tot-f[-1])%mod
```

# 2.单调栈优化dp

# 3.单调队列优化dp

经典划分型dp,使用单调队列进行优化

```python
#注意单调队列优化dp时往往要写滑动窗口
mod=1_000_000_007
fmax=lambda x,y:x if x>y else y
fmin=lambda x,y:x if x<y else y
class Solution:
    def countPartitions(self, nums: List[int], k: int) -> int:
        n=len(nums)
        f=[1]+[0]*n #f[i+1]表示0~i范围划分方案数
        #q_mx,q_mn分别是单调递减队列,单调递增队列,维护滑动窗口最大,最小值
        q_mx,q_mn=deque(),deque()
        l,s=0,0 #l表示窗口左边界,s表示f[l]~f[i]范围累加和,注意当前下标i对应的f为f[i+1]
        for i,x in enumerate(nums):
            s=(s+f[i])%mod #更新s
            #更新两个单调队列
            while q_mx and x>=nums[q_mx[-1]]:
                q_mx.pop()
            q_mx.append(i)
            while q_mn and x<=nums[q_mn[-1]]:
                q_mn.pop()
            q_mn.append(i)
            #不定长滑窗,维护窗口内极差<=k
            while nums[q_mx[0]]-nums[q_mn[0]]>k:
                #当左边界为l时,代表l~i范围内的最大值-最小值<=k
                #此时分割点边界为l,分割点两侧分别为0~l-1和l~i
                #0~l-1对应的划分方案数是f[l]
                #如果此时窗口内极差>k,l需要右移,f[l]就取不到了,因此需要从窗口累加和s中减去f[l],同时l+1
                s=(s-f[l])%mod
                l+=1
                #删除单调队列中的过期元素,即下标<l的元素
                if q_mx[0]<l:
                    q_mx.popleft()
                if q_mn[0]<l:
                    q_mn.popleft()
            f[i+1]=s
        return f[-1]
```

思维扩展:使用双端队列+有序表优化

```python
#本质和上面写法一样,不过用双端队列+有序表代替两个单调队列,写起来更简单
mod=1_000_000_007
fmax=lambda x,y:x if x>y else y
fmin=lambda x,y:x if x<y else y
class Solution:
    def countPartitions(self, nums: List[int], k: int) -> int:
        n=len(nums)
        f=[1]+[0]*n #f[i+1]表示0~i范围划分方案数
        q,d=deque(),SortedList()
        l,s=0,0
        for r,x in enumerate(nums):
            s=(s+f[r])%mod
            q.append(r)
            d.add(x)
            while d[-1]-d[0]>k:
                d.remove(nums[q.popleft()])
                s=(s-f[l])%mod
                l+=1
            f[r+1]=s
        return f[-1]
```

# 4.树状数组/线段树优化dp

# 5.字典树优化dp

# 6.矩阵快速幂优化dp

# 7.斜率优化dp

# 8.WQS二分优化dp

# 9.其他优化dp

**3181.执行操作可以获得的最大总奖励II**

位图优化01背包

```python
#3181(https://leetcode.cn/problems/maximum-total-reward-using-operations-ii/)
class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        #想选m,元素和至多是m-1,选了m之后,元素和至多是2m-1
        m=max(rewardValues)
        s=set()
        #如果数组中包含m-1,一定可以达到最大奖励2m-1
        #同时如果有两个不同的数和为m-1,也可以达到2m-1的目标,这可以用两数之和实现
        for x in rewardValues:
            if x in s:
                continue
            if x==m-1 or m-1-x in s:
                return m*2-1
            s.add(x)
        #位图优化01背包
        #正常f[i][j]表示从前i个物品中是否能得到总奖励j
        #为了优化将f[j]压缩到一维变量f中
        #此时f的第i位表示能否得到总奖励j,0代表不能,1代表能
        f=1
        for x in sorted(set(rewardValues)):
            #由于f[j]=f[j]|f[j-v]
            #因此可以将第k位或到第k+v位中
            #我们目标是将所有的位左移k位
            #(1<<x)-1&f取出f的低x位
            #((1<<x)-1&f)<<x将所有的位左移x位
            f|=((1<<x)-1&f)<<x
        #最高位即是答案
        return f.bit_length()-1
```




[TOC]

# 1.术语

1.指针,引用,成员指针

2.xvalue,gvalue,prvalue

3.constexpr

4.结构化绑定:auto [a,b]=...;

5.右值引用与移动语义:T&&,std::move,std::forward 

6.完美转发

7.属性,[[nodiscard]],[[maybe_unused]]

8.函数模版/类模版

9.可变参数模版/折叠表达式

10.SFINAE与类型特征(Type Traits):std::enable_if_t,std::is_integral_v...

11.Concepts(c++20)

12.CRTP(Curious Recurring Template Pattern)

13.编译期计算(constexpr)与static_assert

14.序列式,std::forward_list,std::array

15.关联式:std::multiset,std::multimap

16.字符串:std::string,std::wstring,std::string_view

17.迭代器:输入迭代器,输出迭代器,前向迭代器,...

18.反向迭代器:std::reverse_iterator

19.迭代适配器:std::back_inserter,std::front_inserter...

20.正则表达式:std::regex,std::smatch

21.智能指针

22.自定义分配器

23.对齐控制

24.Boost.MPL/Boost.Hana

25.Boost.MultiArray,xtensor

26.Eigen

27.fmt

28.std::function,std::any...

29.虚函数

30.Mixin,多重继承

31.反射

32.工厂模式,单例模式,策略模式

33.观察者模式,装饰器模式,适配器模式,命令模式,状态模式,模版方法模式

# 2.基本语法

## 1.vector初始化

```c++
vector<int>cnt(n);
```

## 2.for range循环

```c++
for (int x:nums){
    
}
```

```c++
for (auto& t:tasks){
    
}
```



## 3.多变量赋值

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        //多个相同类型变量的赋值
        ListNode *p=headA,*q=headB;
        while (p!=q){
            //三目表达式
            p=p?p->next:headB;
            q=q?q->next:headA;
        }
        return p;
    }
};
```

## 4.多行字符串

```c++
#include<bits/stdc++.h>
using namespace std;
int main(){
    cout<<"zhe\n"
        "shi\n"
        "yi\n"
        "dao\n"
        "qian\n"
        "dao\n"
        "ti\n";
}
```

也可以用R"()"实现

```c++
#include<bits/stdc++.h>
using namespace std;
int main(){
    cout<<R"(zhe
shi
yi
dao
qian
dao
ti)";
}
```

## 5.格式化输出

### 1.输出%要用%%

```c++
//https://ac.nowcoder.com/acm/contest/18839/1007
#include<bits/stdc++.h>
using namespace std;
int main(){
    int a,b;
    cin>>a>>b;
    printf("%.3lf%%",(double)b/a*100);
    return 0;
}
```

### 2.特殊字符如"要用\进行转译

```c++
//https://ac.nowcoder.com/acm/contest/18839/1008
#include<bits/stdc++.h>
using namespace std;
int main(){
    cout<<"\"Genius is 1% inspiration and 99% perspiration.\""<<endl;
    return 0;
}
```

### 3.保留指定位有效数字

```c++
//https://ac.nowcoder.com/acm/contest/18839/1011
#include<bits/stdc++.h>
using namespace std;
int main(){
    double a;
    cin>>a;
    printf("%.3lf",a);
    return 0;
}
```

### 4.十六进制

```c++
//https://ac.nowcoder.com/acm/contest/18839/1020
#include<bits/stdc++.h>
using namespace std;
int main(){
    int a,b;
    cin>>a>>b;
    printf("%x",a+b);
    return 0;
}
```

### 5.精度

**涉及到除法的问题,最好初始化成浮点数避免精度损失**

```c++
//https://ac.nowcoder.com/acm/contest/18839/1038
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    double a,b,c;
    cin>>a>>b>>c;
    double t=sqrt(a*b*c);
    cout<<4*(t/a+t/b+t/c);
    return 0;
}
```

**涉及到分数时,如三分之一,要写成1.0/3表示浮点数除法**

```c++
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    int n;
    cin>>n;
    printf("%.3lf",3*pow(n,1.0/3));
    return 0;
}
```

### 6.注意输出时要么全用printf,要么全用cout

**混用会出现问题,如果用了关闭同步语句,混用printf,cout,就要在使用printf之后手动刷新缓冲区**

fflush(stdout);

### 7.字符占位

**一个字符占4位**

printf("%4d",x);或cout<<setw(4)<<x;

## 6.字符串处理

### 1.字符串中字符替换

```c++
//C++中std::string是可变的
#include<bits/stdc++.h>
using namespace std;
int main(){
    string s="hello world";
    for(auto &x:s)
        x+=1;
    cout<<s;
    return 0;
}
```

### 2.字符串反转

```c++
https://ac.nowcoder.com/acm/contest/18839/1023
#include<bits/stdc++.h>
using namespace std;
int main(){
    int a;
    cin>>a;
    string s=to_string(a);
    reverse(s.begin(),s.end());
    cout<<s;
    return 0;
}
```

### 3.科学计数法

```c++
#include<bits/stdc++.h>
using ll=long long;
using namespace std;
int main(){
    ll a;
    cin>>a;
    int b=3.156E7;
    cout<<a*b;
    return 0;
}
```

### 4.INT最大值和最小值

INT_MIN=-2147483648

INT_MAX=2147483647

## 7.API

### 1.最大值

```c++
int mx=ranges::max(nums);
```

### 2.多个数max

```c++
//https://ac.nowcoder.com/acm/contest/18839/1035
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    int a,b,c;
    cin>>a>>b>>c;
    cout<<max({a,b,c});
    return 0;
}
```

### 3.minmax

```c++
//一次比较同时得到最小值和最大值
auto [mn, mx] = std::minmax(9, 4);
```

### 4.lower_bound

```c++
//注意C++lower_bound返回的是迭代器
auto it=lower_bound(arr.begin(),arr.end(),target);
//如果要获得值,则通过解引用方式获取
int v=*it;
//如果要获取下标,则通过减去开头迭代器获取
int i=it-arr.begin();
//如果>=target的值在arr不存在,返回的是arr.end(),即尾迭代器,对应越界位置
```

### 5.upper_bound

```c++
idx=ranges::upper_bound(nums,x)-nums.begin();
```

### 6.vector大小

```c++
a=nums.size()
```

### 7.前缀和

```c++
//nums.begin(),nums.end()表示前缀和计算的开头,结尾,最后的nums.begin()表示前缀和输出的数组开头
partial_sum(nums.begin(),nums.end(),nums.begin());
```

### 8.商和余数

```c++
//div来自<cstdlib>
#include<bits/stdc++.h>
using namespace std;
int main(){
    int a,b;
    cin>>a>>b;
    div_t d=div(a,b);
    cout<<d.quot<<" "<<d.rem<<endl;
    return 0;
}
```

### 9.整数转字符串

to_string()

```c++
//https://ac.nowcoder.com/acm/contest/18839/1023
#include<bits/stdc++.h>
using namespace std;
int main(){
    int a;
    cin>>a;
    string s=to_string(a);
    reverse(s.begin(),s.end());
    cout<<s;
    return 0;
}
```

### 10.置位数

```c++
a=popcount((uint64_t)x);
```

### 11.inf

```c++
int ans=INT_MAX;
```

### 12.排序

```c++
//从大到小排序
ranges::sort(nums,greater());
```

### 13.去重

```c++
nums.erase(ranges::unique(nums).begin(),nums.end());
```

### 14.重新分配数组大小

```c++
nums.resize(k);
```

### 15.查找

```c++
VOWEL='aeiou';
VOWEL.find(x)!=string::npos;//查找成功
```

## 8.模拟

```c++
//https://ac.nowcoder.com/acm/contest/19305/1001
//上层金字塔正序遍历,下层金字塔倒序遍历,内部逻辑完全一样,可以直接复用代码
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    int n;
    while (cin>>n){
        for (int i=0;i<n;i++){
            for (int j=0;j<n-i-1;j++){
                cout<<" ";
            }
            for (int j=0;j<2*i+1;j++){
                cout<<"*";
            }
            cout<<endl;
        }
        for (int i=n-2;i>=0;i--){
            for (int j=0;j<n-i-1;j++){
                cout<<" ";
            }
            for (int j=0;j<2*i+1;j++){
                cout<<"*";
            }
            cout<<endl;
        }
    }
    return 0;
}
```

## 9.声明常量

constexpr:声明常量表达式



# 3.STL

## 1.哈希表unordered_map

有关哈希表的使用,这里用两数之和这道题目举例

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        //C++中的哈希表为unordered_map,使用时需要包含<unordered_map>头文件
        unordered_map<int,int>d;
        for (int i=0;i<nums.size();i++){
            //当调用指定键时,C++返回的是迭代器,指向一个键值对
            auto it=d.find(target-nums[i]);
            //如果键不存在,C++默认返回d.end(),即哈希表的尾迭代器
            if (it!=d.end()){
                //如果想访问键对应的值,需要调用it->second
                //注意:it->first代表的是键,it->second代表的才是值
                return {it->second,i};
            }
            d[nums[i]]=i;
        }
        //力扣检查很严格,最后必须有返回值,否则编译不通过
        //也可以去掉这行语句,将循环中的i<nums.size()去掉,因为必定可以返回
        return {};
    }
};
```

哈希表套有序表

```c++
unordered_map<int,set<pair<int,int>>>d;
```

键为二元组的哈希表,将二元组状压成一个long long数从而避免写哈希

```c++
//定义哈希表
unorderd_map<long long,int>e;
//插入元素
e[1LL*shop<<32|movie]=price;
```

## 2.优先队列priority_queue

```c++
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        //比较器,小根堆写法
        auto cmp=[](const ListNode*a,const ListNode*b){
            return a->val > b->val;
        };
        //定义小根堆
        priority_queue<ListNode*,vector<ListNode*>, decltype(cmp)>pq;
        for (auto head:lists){
            if (head){
                pq.push(head);
            }
        }
        ListNode dummy{};
        auto cur=&dummy;
        while (!pq.empty()){
            //访问堆顶元素
            auto node=pq.top();
            //注意pop()返回值为空,因此必须先用top()获取堆顶元素
            pq.pop();
            if (node->next){
                pq.push(node->next);
            }
            cur->next=node;
            cur=cur->next;
        }
        return dummy.next;
    }
};
```

## 3.集合set和unordered_set

虽然unordered_set均摊复杂度是O(1),但是最坏是O(n),会被hack,除非自定义哈希,因此一般使用set

查找:s.find(x),返回的是迭代器

清除:clear()

插入:s.insert(x)

查看大小:s.size()

```c++
//表内元素为二元组的有序表
set<pair<int,int>>a;
//表内元素为三元组的有序表
set<tuple<int,int,int>>b;
//遍历set
for (auto& [_,shop]:a){
    
}
//插入元素
a.emplace(price,shop);
b.emplace(price,shop,movie);
//删除指定元素
a.erase({price,shop});
b.erase({price,shop,movie});
```

## 4.双端队列deque

弹出队尾:q.pop_back()

弹出队头:q.pop_front(),注意返回值为空,如果查询队头需要用q.front()

加入队尾:q.push_back(),注意返回值为空,如果查询队尾需要用q.back()

加入队头:q.push_front()

查询队头:q.front()

查询队尾:q.back()

查询队列是否为空:q.empty()

队列长度:q.size()

## 5.multiset

multiset底层用红黑树实现,允许重复元素

插入:s.insert(v)

查找:it=s.find(v),返回的是迭代器

查找最大值:mx=*s.rbegin()

查找最小值:mn=*s.begin()

删除指定值的第一个元素s.erase(s.find(v))

删除指定值的所有元素:s.erase(v)

清空:s.clear()


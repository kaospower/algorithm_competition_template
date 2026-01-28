[TOC]

# 1.字符串API

**和字符串相关的题目,使用api可以极大降低代码长度**

## 1.replace

**替换字符**

```python
#1323(https://leetcode.cn/problems/maximum-69-number/description/)
class Solution:
    def maximum69Number(self, num: int) -> int:
        return int(str(num).replace("6", "9", 1))
```

## 2.isdigit/isalpha

**判断字母/数字**

```python
#面试题17.05(https://leetcode.cn/problems/find-longest-subarray-lcci/description/)
#s.isdigit()判断数字,s.isalpha()判断字母
class Solution:
    def findLongestSubarray(self, array: List[str]) -> List[str]:
        s=0
        d=defaultdict(int)
        l,r=0,0
        maxv=0
        for i,x in enumerate(array):
            if s not in d:
                d[s]=i-1
            s+=1 if x.isdigit() else -1
            if s in d and i-d[s]>maxv:
                maxv=i-d[s]
                l,r=d[s],i
        return array[l+1:r+1]
```

## 3.isalnum

**如果一个字符是数字或者是字母,该函数返回True**

```python
s=[x for x in s.lower() if x.isalnum()]
```

## 4.lstrip/rstrip

**去除首尾相同项**

```python
#2211(https://leetcode.cn/problems/count-collisions-on-a-road/)
class Solution:
    def countCollisions(self, s: str) -> int:
        s = s.lstrip('L')  
        s = s.rstrip('R')  
        return len(s) - s.count('S') 
```

## 5.startswith

**判断前缀**

```python
#1268(https://leetcode.cn/problems/search-suggestions-system/)
#a.startswith(b),判断b是否是a的前缀
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        m1,m2,m3,target='{','{','{',''
        ans,tmp=[],[]
        n=len(searchWord)
        for i in range(n):
            target+=searchWord[i]
            m1,m2,m3='{','{','{'
            tmp=[]
            for product in products:
                if product.startswith(target):
                    if product<m1:
                        m3,m2,m1=m2,m1,product
                    elif product<m2:
                        m3,m2=m2,product
                    elif product<m3:
                        m3=product
            if m1!='{':
                tmp.append(m1)
            if m2!='{':
                tmp.append(m2)
            if m3!='{':
                tmp.append(m3)
            ans.append(tmp[:])
        return ans
```

## 6.string.ascii_lowercase/string.ascii_uppercase

**string.ascii_lowercase可以枚举小写字母表**

**string.ascii_uppercase可以枚举大写字母表**

## 7.swapcase()

**x.swapcase()可以将小写字母转化成大写,大写转化成小写**

也可以用ascii码值异或32交换大小写

## 8.dict

**d=dict(zip(ascii_lowercase,score))可以将元组转化成字典**

## 9.find

**寻找子串开头位置,i=b.find(a),a在b中开头位置**

```python
#28(https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)
def strStr(self, haystack: str, needle: str) -> int:
    return haystack.find(needle)
```

## 10.eval

eval()可用于表达式解析,也可以用来计算算术表达式的值

## 11.upper/lower

**s.upper()将字符串中所有字母转化成大写,s.lower()将字符串中所有字母转化成小写**

## 12.isupper/islower

检验字符串是否全是大写/小写

## 13.capitalize

将字符串第一个字母如果是小写变成大写,其他是字母的字符变成小写

```python
s=s.capitalize()
```



# 2.遍历API

## 1.chain.from_iterable

**将二维数组化成一维**

```python
class Solution:
    def clearStars(self, s: str) -> str:
        st=[[] for _ in range(26)]
        for i,x in enumerate(s):
            if x!='*':
                st[ord(x)-ord('a')].append(i)
            else:
                for y in st:
                    if y:
                        y.pop()
                        break
        #chain.from_iterable(st)遍历st的每个数组,然后将结果拼成一个数组
        return ''.join(s[i] for i in sorted(chain.from_iterable(st)))
```

## 2.accumulate

**accumulate除了计算前缀和,还可以传入max,min,gcd等计算前缀信息**

```python
#3502(https://leetcode.cn/problems/minimum-cost-to-reach-every-position/)
class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        return list(accumulate(cost,min))
```

## 3.reduce

**reduce对数组中的元素迭代进行某种运算**

```python
#2683(https://leetcode.cn/problems/neighboring-bitwise-xor/)
class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        return reduce(xor,derived)==0
```

## 4.pairwise

**数组中的相邻元素两两配对**

```python
#3423(https://leetcode.cn/problems/maximum-difference-between-adjacent-elements-in-a-circular-array/description/)
class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        return max((abs(x-y))for x,y in pairwise(nums+[nums[0]]))
```

# 3.工程API

## 1.cache_clear

**清除缓存**

```python
max=lambda x,y:x if x>y else y
class Solution:
    def minCost(self, nums: List[int]) -> int:
        n=len(nums)
        @cache
        def f(i,j):
            if i==n:
                return nums[j]
            if i==n-1:
                return max(nums[j],nums[i])
            a,b,c=nums[j],nums[i],nums[i+1]
            return min(f(i+2,i+1)+max(a,b),\
                       f(i+2,i)+max(a,c),\
                       f(i+2,j)+max(b,c))
        ans=f(1,0)
        #需要清除缓存,防止爆内存
        f.cache_clear() 
        return ans
```

## 2.json.dumps

**打印defaultdict**

```python
# 利用递归构造一个默认值为多叉树的字典
def tree():
    return defaultdict(tree)

# 创建树的根节点
multi_tree = tree()

# 添加节点：例如，构建一棵简单的树
multi_tree['root']['child1']['child1_1'] = "leaf1"
multi_tree['root']['child1']['child1_2'] = "leaf2"
multi_tree['root']['child2']['child2_1'] = "leaf3"

# 如果想看到结构,可以转换为普通字典打印(注意:转换只适用于调试，不支持递归无限转换),以json形式打印树
print(json.dumps(multi_tree, indent=4))
```

## 3.tuple

**将不可哈希对象list转为可哈希对象**

```python
#枚举
class Solution:
    def countDistinct(self, nums: List[int], k: int, p: int) -> int:
        s, n = set(), len(nums)
        for i in range(n):
            cnt = 0
            for j in range(i, n):
                if nums[j] % p == 0:
                    cnt += 1
                    if cnt > k:
                        break
                s.add(tuple(nums[i: j + 1])) #tuple
        return len(s)
```

# 4.数据结构API

## 1.deque(maxlen=size)

**指定容量的队列**

```python
#346(https://leetcode.cn/problems/moving-average-from-data-stream/)
class MovingAverage:

    def __init__(self, size: int):
        self.d = deque(maxlen=size)

    def next(self, val: int) -> float:
        self.d.append(val)
        return sum(self.d) / len(self.d)
```

## 2.list

### 1.insert

### 2.a[:0]=[v]

### 3.a=[v]+a

以上三种写法都是向开头添加元素

a.insert(0,v),效率最高

a[:0] = [v],效率略低

a=[v]+a,效率最低

```python
#1670(https://leetcode.cn/problems/design-front-middle-back-queue/description/)
class FrontMiddleBackQueue:

    def __init__(self):
        self.q = []

    def pushFront(self, val: int) -> None:
        #这两种方法都可以向列表头添加元素
        self.q[:0] = [val]
        # self.q.insert(0,val)

    def pushMiddle(self, val: int) -> None:
        # 注意正确计算位置
        pos = len(self.q) // 2
        self.q[pos:pos] = [val]

    def pushBack(self, val: int) -> None:
        self.q.append(val)

    def popFront(self) -> int:
        if not self.q:
            return -1
        ret = self.q[0]
        self.q[0:1] = []
        return ret

    def popMiddle(self) -> int:
        if not self.q:
            return -1
        # 注意正确计算位置
        pos = (len(self.q) - 1) // 2
        ret = self.q[pos]
        self.q[pos:pos+1] = []
        return ret

    def popBack(self) -> int:
        if not self.q:
            return -1
        return self.q.pop()
```

### 4.index

查找值的索引,可以在arr中查找x第一次出现的位置

## 3.set

### 1.add

**使用 `add()` 方法可以向集合中添加一个元素。**

### 2.remove

**使用 `remove()` 方法可以删除指定的元素，如果该元素不存在，会抛出 `KeyError`。**

### 3.discard

**使用 `discard()` 方法也可以删除元素，但如果元素不存在，则不会抛出错误。**

### 4.update

**使用 `update()` 方法可以添加多个元素。**

### 5.pop

**`pop()` 方法会随机移除并返回集合中的一个元素。如果集合为空，调用 `pop()` 会引发 `KeyError`。**

```python
my_set = {1, 2, 3}
my_set.add(4)
print(my_set)  # 输出: {1, 2, 3, 4}

my_set.remove(2)  # 删除 2
print(my_set)  # 输出: {1, 3, 4}

my_set.discard(5)  # 不会抛出错误
print(my_set)  # 输出: {1, 3, 4}

my_set.update([5, 6])
print(my_set)  # 输出: {1, 3, 4, 5, 6}

my_set = {1, 2, 3, 4}

# 随机移除一个元素
removed_element = my_set.pop()
print(f"Removed element: {removed_element}")
print(f"Remaining set: {my_set}")
```

## 4.dict

### 1.del

### 2.pop

`del dict[key]` 用于删除指定的键及其对应的值。如果键不存在，会抛出 `KeyError`。

`del` 的性能通常比 `pop` 更高，因为它直接从字典中删除键值对，而不返回任何值。



`dict.pop(key)` 用于删除指定的键，并返回该键的值。如果键不存在，可以通过提供默认值来避免 `KeyError`，例如 `dict.pop(key, default)`。

`pop` 在删除键的同时还返回该键的值，因此相对 `del` 来说，它的开销略高。

使用 `del` 时更快且不返回值。

使用 `pop` 时可以同时获取被删除的值，适合需要用到该值的情况。

### 3.setdefault

**`setdefault` 是 Python 字典（`dict`）的方法，用于获取指定键的值。如果该键不存在，则会将其添加到字典中，并设置一个默认值（通常是一个空集合或其他类型）。这在需要初始化字典中的值时非常有用。**

## 5.heapq

`heapq` 是 Python 的标准库模块，用于实现堆（heap）算法，提供了优先队列的功能。它支持最小堆（即堆顶元素为最小值），可以用于高效地处理优先级队列和其他需要频繁获取最小值的场景。

**`heapq.heappush(heap, item)`**：将元素 `item` 添加到堆中，维护堆的性质。

**`heapq.heappop(heap)`**：从堆中弹出并返回最小元素，同时维护堆的性质。调用此方法会修改原始堆。

**`heapq.heapify(x)`**：将列表 `x` 转换为堆，原地转换，时间复杂度为 O(n)O(n)O(n)。

**`heapq.heappushpop(heap, item)`**：将 `item` 推入堆中，然后弹出并返回最小元素。此方法比单独使用 `heappush()` 和 `heappop()` 更高效。

**`heapq.nlargest(n, iterable, key=None)`**：返回可迭代对象 `iterable` 中的前 `n` 个最大元素，结果以列表形式返回。

**`heapq.nsmallest(n, iterable, key=None)`**：返回可迭代对象 `iterable` 中的前 `n` 个最小元素，结果以列表形式返回。

```python
import heapq

# 创建一个空堆
heap = []

# 添加元素
heapq.heappush(heap, 10)
heapq.heappush(heap, 5)
heapq.heappush(heap, 20)
heapq.heappush(heap, 15)

print("堆的状态:", heap)  # 输出: [5, 10, 20, 15]

# 弹出最小元素
min_element = heapq.heappop(heap)
print("弹出的最小元素:", min_element)  # 输出: 5
print("堆的状态:", heap)  # 输出: [10, 15, 20]

# 转换列表为堆
nums = [3, 1, 4, 1, 5, 9]
heapq.heapify(nums)
print("堆化后的列表:", nums)  # 输出: [1, 1, 4, 3, 5, 9]

# 获取前两个最大元素
largest_two = heapq.nlargest(2, nums)
print("前两个最大元素:", largest_two)  # 输出: [5, 4]

# 获取前两个最小元素
smallest_two = heapq.nsmallest(2, nums)
print("前两个最小元素:", smallest_two)  # 输出: [1, 1]
```

## 6.deque

`deque`（双端队列）是 Python 中 `collections` 模块提供的一种数据结构，它允许在两端快速添加和删除元素。与列表相比，`deque` 在队首和队尾的插入和删除操作的时间复杂度为 O(1)，而列表在队首的操作复杂度为 O(n)，因此在需要频繁在两端进行操作时，`deque` 是更优的选择。

-   **`append(x)`**：在队尾添加元素 `x`。
-   **`appendleft(x)`**：在队首添加元素 `x`。
-   **`pop()`**：从队尾移除并返回一个元素。
-   **`popleft()`**：从队首移除并返回一个元素。
-   **`extend(iterable)`**：在队尾添加来自可迭代对象的元素。
-   **`extendleft(iterable)`**：在队首添加来自可迭代对象的元素（注意，元素的顺序会被反转）。
-   **`rotate(n)`**：将队列中的元素旋转 `n` 个位置，正值表示向右旋转，负值表示向左旋转。
-   **`clear()`**：清空队列。
-   **`count(x)`**：返回 `x` 在队列中出现的次数。
-   **`remove(value)`**：删除队列中第一次出现的值。

```python
from collections import deque

# 创建一个空的 deque
dq = deque()

# 添加元素
dq.append(1)
dq.append(2)
dq.append(3)
print("Deque after appending:", dq)  # 输出: deque([1, 2, 3])

# 从队首添加元素
dq.appendleft(0)
print("Deque after appending left:", dq)  # 输出: deque([0, 1, 2, 3])

# 从队尾移除元素
removed = dq.pop()
print("Removed from right:", removed)  # 输出: 3
print("Deque after popping:", dq)  # 输出: deque([0, 1, 2])

# 从队首移除元素
removed_left = dq.popleft()
print("Removed from left:", removed_left)  # 输出: 0
print("Deque after popping left:", dq)  # 输出: deque([1, 2])

# 扩展 deque
dq.extend([4, 5])
print("Deque after extending:", dq)  # 输出: deque([1, 2, 4, 5])

# 旋转 deque
dq.rotate(1)  # 向右旋转一个位置
print("Deque after rotation:", dq)  # 输出: deque([5, 1, 2, 4])
```

## 7.PriorityQueue

`queue.PriorityQueue` 是 Python 标准库中的一个线程安全的优先队列实现。它基于堆结构，可以在多线程环境中安全地使用，适合于需要优先级处理的场景。以下是 `PriorityQueue` 的详细介绍和使用示例。

-   **`put(item)`**：将一个元素添加到优先队列中。可以传入一个元组，通常是 `(priority, item)`，其中 `priority` 是优先级，`item` 是要存储的值。
-   **`get()`**：从队列中提取并返回优先级最高的元素（即最小优先级）。如果队列为空，会阻塞直到有元素可用。
-   **`empty()`**：检查优先队列是否为空，返回布尔值。
-   **`qsize()`**：返回优先队列中元素的数量。

```python
from queue import PriorityQueue
import threading
import time

# 创建一个优先队列
pq = PriorityQueue()

# 定义一个生产者线程
def producer():
    for i in range(5):
        pq.put((i, f"task {i}"))  # 使用 (priority, task) 元组
        print(f"Produced: task {i}")
        time.sleep(1)

# 定义一个消费者线程
def consumer():
    while True:
        priority, task = pq.get()
        print(f"Consumed: {task} with priority {priority}")
        if priority == 4:  # 假设当消费到 task 4 时结束
            break
        time.sleep(2)

# 启动线程
producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()
```



# 5.魔法函数API

**很多时候需要手写比较器,掌握魔法函数API就很重要**

## 1.\__lt__

**比较器**

```python
#23(https://leetcode.cn/problems/merge-k-sorted-lists/)
#手写比较器,让堆可以比较节点大小
ListNode.__lt__ = lambda a, b: a.val < b.val 

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        cur = dummy = ListNode()  
        h = [head for head in lists if head]  
        heapify(h)  
        while h:  
            node = heappop(h)  
            if node.next: 
                heappush(h, node.next)  
            cur.next = node  
            cur = cur.next  
        return dummy.next  
```

# 6.解包运算符*

a,*rest = data

# 7.numpy API

## 1.np.convolve

**多项式卷积(FFT)**

```python
#3549(https://leetcode.cn/problems/multiply-two-polynomials/)
import numpy as np
class Solution:
    def multiply(self, poly1: List[int], poly2: List[int]) -> List[int]:
        return np.convolve(poly1, poly2).tolist()
```

# 8.数学API

## 1.atan2

返回反正切值,且值控制在[-pi,pi]

```python
res=atan2(8,5)
```

## 2.pi

圆周率常量

# 9.其他常用API

## 1.random.choice

**选择随机元素**

```python
import random

my_list = [1, 2, 3, 4, 5]
random_item = random.choice(my_list)
print(random_item)  # This will print a random item from my_list
```

## 2.sort

**注意python自带的sort()实现是基于归并排序,因此遇到类似合并两个有序表的题目,直接调用sort就行,不要再手写归并排序了**

## 3.groupby

**groupby(arr)将arr中连续的相同值分组,返回值有两个,第一个是这个值,第二个是这些相同值的迭代器**

## 4.starmap

**starmap和map相比,可以对可迭代对象进行解包**

```python
A = sum(starmap(eq, zip(secret, guess)))
```

## 5.&交集

```python
#两个Counter对象取交集,key为两者都有的键,val为这个键在两者中对应的值中较小的那个
B = sum((Counter(secret) & Counter(guess)).values()) - A
```

## 6.Counter/elements

**Counter计数,elements用计数得到的结果还原原来的list**

## 7.-|&差并交

**集合的差并交操作**

```python
a={4,5,6}
b={6,7,8}
c=a-b #差集
c=a|b #并集
c=a&b #交集
```

## 8.input

**输入操作**

```python
#流式输入
input = sys.stdin.read
data = input().strip().splitlines()
```

## 9.next

**next取出迭代器中下一个元素**

```python
#用迭代器寻找第一个满足条件的坐标
i,j=next((i,j) for i in range(n) for j in range(m) if grid[i][j])
#等价于
# for i in range(n):
#     for j in range(m):
#         if grid[i][j]:
#             return (i, j)
```

## 10.切片

```python
s[a:b:c]#表示从a到b,左闭右开切片,步长为c
s[a:b:-1]#步长为负数,表示从右向左反向切片,此时a,b必须写成负数
s[::c]#表示从0到n+1以c步长切片,n为s长度,即下标越界位置
s[::-1]#表示从-1到-(n+1)以步长1反向切片,-(n+1)为负数下标越界位置,即反转字符串
```

## 11.nlargest

```python
#取出数组中最大的三个值,注意nlargest底层调用的堆,可以对各种结构使用
a=nlargest(3,arr) 
```

## 12.nsmallest

```python
#求数组前k小,注意这个方法底层是堆实现,但是优化较好,当k=2或3时比手写迭代要快
a=nsmallest(k,arr)
```


1.最大值

```go
slices.Max(nums)
```

2.声明数组

```go
cnt := make([]int, mx+1)
```

3.普通for循环

```go
for i:=0;i<n;i++{
    
}
```

4.for range循环

```go
for _, x := range nums {
	cnt[x]++
}
```

5.二分api,查找>=x的第一个位置

```go
sort.SearchInts(nums,int(x))
```

6.置位数

```go
bits.OnesCount(uint(x))
```

7.inf

```go
ans:=math.MaxInt
```

8.排序

```go
//由大到小
slices.SortFunc(nums,func(a,b int) int {return b-a})
```

9.去重

```go
nums=slices.Compact(nums)
```


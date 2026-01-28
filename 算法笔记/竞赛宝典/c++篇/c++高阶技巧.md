C++泛型编程和STL技术

# 1.模版

模版:建立通用的模具,大大提高复用性

## 1.函数模版

```c++
template<typename T>//声明模版
void mySwap(T &a,T &b)
{
    T temp=a;
    a=b;
    b=temp;
}
```

## 2.类模版


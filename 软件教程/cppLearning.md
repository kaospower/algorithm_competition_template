1.

**chap2**

```c++
extern
  
  
/*extern keyword is used to declare a variable*/
/*this is used when you want to call a variable in another file*/
```

```c++
::

#include<iostream>
int reused=10; //this is a global variable
int main(){
  std::cout<<reused<<endl;//the output is 10
  int reused =45;
  std::cout<<reused<<endl;//the output is 45, local variable override the global variable
  
  std::cout<<::reused<<endl;
  /*output is 10, we use :: without left operand to call a global variable */
  return 0;
}
```

```c++
//& 用法
//1. reference
int &r=i
//2. get the address
int *p=&i
//*用法
//1.定义指针
int *p=&a
//2.解除引用(dereference)
*p=w
  
/*reference is not an object*/
```

```c++
//声明空指针的三种方法
int *p1=nullptr
int *p2=0
int *p3=NULL
```

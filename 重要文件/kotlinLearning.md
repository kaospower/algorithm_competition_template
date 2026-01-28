**How to create a new kotlin project in IDEA**

when you click new project, select the kotlin button of the left side bar, then choose Native Application button in the middle, then type your project name and press Next, you finish the creating of a kotllin project.

Remember your Project JDK should be jdk 11.

![avatar](/Users/kaosdragon/Desktop/readingNotes/kotlinNewProject.jpeg)



1.Kotlin文件名: *.kt

2.Kotlin helloworld:

```kotlin
//Kotlin语言面向函数编程,main方法不需要被包含在class中,这点与java不同
fun main(){
  println("hello world!")
}
```

3.Kotlin variable and string:

```kotlin
var a:Int=1
var b:String="apple"
println(b[4]) //输出应为e,Kotlin可根据字符串下标索引相应字符
println(b.get(4)) //等价写法
for(i in b){
  println(i)
} //输出应为a,p,p,l,e
println(b.toSortedSet()) //输出为a,e,l,p  按照字母表顺序打印,重复字符只保留一个,因为返回的是集合

var s:String="orange$a juice" //打印s应为orange1 juice
var s2:String="orange${a} juice" //等价写法

var c:String="""
	hello
	world
	!
""".trimIndent()  //多行字符串,trimIndent去除锁进
```

4.Kotlin array:

```kotlin
var a:DoubleArray = doubleArrayOf(1.5,2.4,5.6)
var b:Array<Int> = arrayOf(1,2,3)
val c= arrayOf<Int>() //val表示声明一个常量
val d:IntArray=IntArray(10) //创建一个长度为10的数组,元素初始化为0
```

5.Kotlin list:

```kotlin
val list:List<Int> = listOf() //新建空list
val list:List<Int> = listOf(1,2,3) //不可变list
val list2:MutableList<Int> = mutableListOf(1,2,3) //可变list
list2.add(5) //在list末尾添加元素	
list2.add(0,44) //在list下标为0位置处添加元素
list2.remove(1) //清除list下标为1处的元素
list2.clear()  //清除list
list.isEmpty() //检查list是否为空
```

6.Kotlin Map:

```kotlin
val map = mutableMapOf("123" to 1)
val map2 = mapOf("456" to 2)
println(map2.get("456"))
println(map2["456"]) //等价写法
```

7.Kotlin Pair:

```kotlin
val pair=Pair<String,Int>("123",1) //创建元组
println(pair.first) //打印元组的第一个元素 123
println(pair.second) //打印元组的第二个元素 1
println(pair.toString()) //结果为(123,1)
```

定义自己的元组

```kotlin
fun main(){
    data class MPair<A,B>(
        val first:A,
        val second:B,
    ){
        override fun toString(): String {
            return "$first - $second"
        }
    }
    
    val pair=MPair<String,Int>("123",1)
    println(pair.first) //输出123
    println(pair.second) //输出1
    println(pair.toString()) //输出123 - 1
}
```

三元组:

```kotlin
val triple = Triple<String,Int,Float>("123",1,1.2f)
```

8.条件语句

```kotlin
val b=1
val res:Int=if(b==1){
    4
}else if(b==3){
    2
}else{
    1
}

println(res) //输出为4
```

9.when

涉及多个条件选择时用when

单纯的循环用while

```kotlin
val b=1
when(b){
    1,2->{
        println("this is 1")
    }
    else ->{
            
    }
}	//输出this is 1
```



```kotlin
val b:Any=1
when(b){
    is Int ->{
        println("this is 1")
    }
    is String ->{
        println("this is a string")
    }
    else ->{

    }
} //输出this is 1

```



```kotlin
val b:Any=1
when(b){
    in 1 ->{
        println("this is 1")
    }
    is String ->{
        println("this is a string")
    }
    else ->{

    }
} //输出this is 1
```



```kotlin
val b:Any=1
when(b){
    in 2..10 ->{ //区间写法
        println("this is 1")
    }
    is Int ->{
        println("this is int")
    }
    else ->{

    }
} //输出this is int
```

10.三元表达式

```kotlin
val b: Any = 1
val res: Boolean = if (b == 1) true else false
println(res) //输出true
```

11.区间

```kotlin
val range1:IntRange =(1..10) //1到10闭区间
val range2:IntRange =(1 until 10) //1到10左闭右开区间
val range3:IntProgression =(10 downTo 1)
val range4:IntProgression =(1..10 step 2)
val a=5
println(a in range1) //输出为true,通过in关键字来使用区间
```

12.for循环

```kotlin
val arr:Array<Int> = arrayOf<Int>(14,24,53)
    for (i:Int in arr) {
        println(i)
    } //输出数组元素的值  14,24,53

    for (i:Int in arr.indices) {
        println(i)
    } //输出索引   0,1,2

    for (i:IndexedValue<Int> in arr.withIndex()) {
        println("${i.index} ${i.value}")
    } //输出索引和值   0 14 , 1 24 , 2 53

		arr.forEach { value ->
        println(value)
    } //增强for循环 输出: 14,24,53
		arr.forEachIndexed { index,i ->
        println("$index $i")
    } //输出下标和值 输出: 0 14 , 1 24 , 2 53
```

13.while循环

```kotlin
val range:IntRange =0..3
var a=1
while (a in range){
    a++
    println("in the range")
}
```

14.break

```kotlin
a@ for (i:Int in 0..4){   
    for (j:Int in 0..4){
        if(i==2 && j==2){
            break@a        //@a为标签
        }
        println("$i $j")
    }
}
println("finish")
```

15.return

```kotlin
(0..10).forEach() a@{   it->  //lambda表达式
    if(it==2){
        return@a
    }
    println(it)
} //输出为 1,3,4,5,6,7,8,9,10

run a@{         //run为kotlin的一个高阶函数
    (0..10).forEach() {   it->
        if(it==2){
            return@a   //退出最外层循环
        }
        println(it)
    }

} //输出为0,1
```

16.函数

```kotlin
fun a():Unit{   //Unit表示返回值为空类型
  
}

fun a2():Int=1 //只有一段语句可以省略花括号
fun a3()=1 //最简版,利用推导特性


fun a4(b:()->Unit){
    println(b::class.qualifiedName) //打印变量类型
}
val a:Unit=a4{}
println(a)  //输出结果: null, kotlin.Unit


fun<T>a(b:()->Unit,c:Int?=null){ //<T>范型, ?表示变量可能为空
    val i:Int =c!! +1 //!!表示该变量绝对不能为空
}


//多参函数
fun main() {
    val a:Unit=a(1,2,3)
}

fun a(vararg s:Int){
    s.forEach {
        println(it)
    }
}

//另外一个例子
fun main() {
    val a:Unit=a(s= intArrayOf(1,2,3),b=1)
}

fun a(c:Int=10,vararg s:Int,b:Int){
    s.forEach {
        println(it)
    }
}
```

17.类与对象

```kotlin
fun main(){
    val a=A()
    a.b=3
    println(a.b)
}

class A{
    var b:Int=1
    get() = field+1  //field指b
    set(value){
        println(value)  //value指赋值变量
        field-=1
    }
}

//常量不能用set方法,属性的值(b)的修饰符(public,private等)必须与get()方法的修饰符相一致
```

18.Kotlin构造函数

```kotlin
fun main(){

}

class A(age:Int,name:String){
    private val age: Int
    private val name: String
    constructor(age:Int):this(age,"Tom") //次构造函数代理
    init {
        this.age=age
        this.name=name
    }


}
```

```Kotlin
fun main(){
    val a=A("Peter") //调用次构造函数2
}

class A(age:Int,name:String){
    private val age: Int
    private val name: String
    constructor(age:Int):this(age,"Tom") //次构造函数1调用主构造函数
    constructor(name:String):this(1) //次构造函数2调用次构造函数1
    init {
        this.age=age
        this.name=name
    }



}
```

```kotlin
fun main(){
    val a=A(age=null,"Peter")
    a.printInfo()
}

class A(var age:Int?=null,var name:String){      //全局版的构造函数
    fun printInfo(){
        println("$name $age")
    }



}
```

19.权限修饰符和继承

```kotlin
fun main(){
    
}

open class A (){
    protected fun print(){
        
    }

}

class B:A(){
    fun print2(){
        print()
    }
}
```

```kotlin
abstract class A{ //抽象类,不能被实例化,只能被继承

}

final class A{  //不能被继承
    
}

open class A{  //open使得可以继承
  open fun print(){
    
  }
}
```

```kotlin
open class A{
    open fun print(){  //不加open下面重写会报错

    }
}

class B:A(){
    override fun print(){
        super.print()
    }
}

enum class Menu{  //枚举类
    
}

class A{   
    var b=1
    inner class B{    //内部类
        fun print(){
            this@A.b
        }
    }
}

```

20.接口

```kotlin
fun main(){
    A().test(object :Callback{
        override fun finish() {
            TODO("Not yet implemented")
        }
    })
    
    
}

class A{
    fun test(callback: Callback){
        callback.finish()
    }
}

interface Callback{
    fun finish()
}
```

21.

```kotlin
var myString: String? = null
```

?表示允许变量值为空

22.

elvis operator

```kotlin
x=savedInstanceState?.getInt("x")?:2
```

如果冒号前面的值不为空则返回前面的值，否则返回冒号后面的值



**Create an array**

```kotlin
//define an string type array and it can not be change because we use val keyword.
val options=arrayOf("Rock", "Paper", "Scissors")
```

**define a function**

```kotlin
fun max(a:Int,b:Int):Int{ //kotlin function can have not return value
  //if you want the function to return nothing,else use :Unit or just write nothing
  val maxValue=if(a>b) a else b // this writing is called if expression
  return maxValue
}
```

more simple way of writing code above:

```kotlin
fun max(a:Int, b:Int):Int=if(a>b) a else b 
//this requires function has a single expression


//or even more simple
fun max(a:Int, b:Int)=if(a>b) a else b 
//we can write in this way because the compiler can infer the return type
```


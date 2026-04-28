lesson2

1.activity

```xml
<activity>
  <indent-filter>
  </indent-filter>
</activity>
```

2.view: a view is an object that draws something on the screen that the user can interact with

3.dp:density-independent pixels

```kotlin
private lateinit var nameET:EditText
```

!!: not-null assertion operator

4.初始MainActivity.kt代码

```kotlin
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle

class MainActivity : AppCompatActivity() {
		override fun onCreate(savedInstanceState: Bundle?) {
				super.onCreate(savedInstanceState)
				setContentView(R.layout.activity_main)
    }
}
```

(1)onCreate():

all activities have to implement the onCreate() method. This method gets called when the activity object gets created, and it's used to perform basic setup such as what layout the activity is associated with.

This is done via a call to setContentView.

R.layout.activity_main tells Android this activity uses activity_main.xml as its layout
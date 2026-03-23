Agent Skills(https://www.bilibili.com/video/BV162cPzhEGU/)  
Prompt:提示词  
结构化提示词:详细规定步骤的提示词  
Command:用短命令替换一段固定内容的功能  
System Prompt(系统提示词)  
System Prompt比User Prompt(用户提示词)效果更好  
Metadata(元数据):提示型的数据  
References和Scripts:  
按需加载数据的读取方式,渐进式披露  

Skills  
把常用的用户提示词变成了一段段以文件形式存在的系统提示词  
再通过Metadata和拆分文件,实现按条件和场景加载数据,大大减少token的消耗量  
并将提示词可能用到的参考资料和代码,分别放入References和Scripts文件夹  
一个负责读取参考资料,一个负责跑代码  
将提示词文件主入口改名为skill.md,再将它们共同打包为一个文件夹,给个命名  
这个被外化为文件夹形式存在的且可动态加载的系统提示词其实就是所谓的skill  

skill完整工作流程  
将它放到Claude code的skills目录底下,这样就算完成了安装  
此时ClaudeCode就能识别到它的命令  
然后我们像往常一样在聊天框里发送自己想做的事情  
ClaudeCode就会加载本地的多个skill文件,将它们的metadata一起发给大模型  
大模型识别返回当前需要哪个skill,告诉ClaudeCode  
ClaudeCode加载对应skill文件到系统提示词里,发给大模型  
大模型根据需要让Claude code依次读取可能需要参考的多份资料,甚至是执行本机代码脚本生成pdf  
并将结果给到大模型,大模型最终输出完整结果给用户,完成整个流程  

Skill和MCP的区别  
MCP协议  
MCP插件  
skills:操作经验,规定在什么场景下按什么顺序组合使用哪些工具  

Skills和Workflow的区别   
n8n,通过拖拉拽的方式,快速构建一条流水线  
这种通过规则配置把多个步骤进行编排和调度的流程就叫Workflow  
skills本质上也是做"逻辑编排"  
workflow的流程结构在设计阶段就确定好了,  
skills的执行流程由大模型驱动,灵活性相对更高,可以理解为大模型驱动的workflow  




---
layout: post
title: Java基础笔记-Java内存区域
description: 关于Java JVM的一些学习笔记
category: blog
---
#一 运行时的数据区组成   
![图示1](/images/blog/java-jvm-store-model.png)
   
1. 程序计数器：当前线程执行的字节码的行号指示器。自己吗解释器通过改变程序计数器(PC)的值来选取下一条需要执行的字节码指令。分支、循环、跳转、线程回复等基础功能都依赖于它。
&nbsp;&nbsp;&nbsp;&nbsp;例如：Java多线程机制。线程轮流切换，分配CPU执行时间，任一时刻，一个CPU只会执行一条线程指令，每个线程都需要一个独立PC，以保证线程切换后能正确恢复。   
2. 虚拟机栈，四点说明：   
<ul>
  <li><1> 生命周期随着线程存亡</li>
  <li><2>Java方法执行的内存模型：每个方法在执行的同时会创建一个线帧，用于存储局部变量表、操作数栈、动态链接、方法出口等信息</li>
  <li><3>每个方法从调用直至执行完成的过程，就对应着一个线帧在虚拟机栈中从入栈到出栈的过程</li>
  <li><4>能发生的两类异常:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;<I>线程的请求深度大于虚拟机所允许的深度，会抛出StackOverflowError<br>
      &nbsp;&nbsp;&nbsp;&nbsp;<II>虚拟机栈可动态扩展时无法申请到足够的内存，就会抛出OutOfMemoryError</li>
</ul>   
3. 本地方法栈：作用与虚拟机相同，不同的是虚拟机栈为虚拟机执行Java方法（即字节码）服务，而本地方法栈为虚拟机执行本地方法（Native）服务。   






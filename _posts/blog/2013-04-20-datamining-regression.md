---
layout:     post
title:      数据挖掘方法之二：回归模型（简单线性回归）
category: blog
description: 数据挖掘专栏
---
##注：文中所使用代码为R   

##一 概念    

简单线性回归模型是用于估计一个连续预测变量和一个连续回应变量的线性关系。回归方程或估计回归方程(estimated regression equation,ERE)： 
<hr>
  <h4 align="center">y~=b0+b1*x </h4>
<hr>
<ul><li>y~是回应变量的估计值</li>  <li>b0是回归线在y轴上的截距</li>  <li>b1是回归线的斜率</li> <li>b0和b1称为回归系数</li></ul>
         
##二 实例   
  
<ul><li>数据来源: [谷物数据集](http://lib.stat.cmu.edu/DASL/Datafiles/Cereals.html)   </li>
<li>数据描述：谷物数据集,包含了77种早餐谷物的16个属性对应的营养信息</li>   
</ul>
首先导入数据：
   
    sugar<-read.table(file="/LabData/RData/regression/nutrition.txt",header=TRUE)
 
部分数据概览如下：
   
    edit（sugar）  

![数据集](/images/blog/regression1.png)
就给定谷物的含糖量对该谷物的营养成分进行评价，77种谷物的营养级别与含糖量的散点图和拟合回归线如下:   

    plot(data=sugar,rating~sugars,main="营养级别和含糖量的散点图及拟合线",xlab="含糖量",ylab="营养级别")  
    lm.reg<-lm(data=sugar,rating~sugars)  
    abline(lm.reg,lty=4,lwd=3)   
    
![拟合](/images/blog/regression2.png)
 
使用线性回归模型拟合结果如下：

    lm(data=sugar,rating~sugars)  
    Call:  
    lm(formula = rating ~ sugars, data = sugar)  
    Coefficients:  
      (Intercept)       sugars    
       59.284       -2.401   

这里给定ERE为 ： y~=59.284-2.401*sugars， 所以b0=59.284,b1=-2.401   
<br>

###误差残留   

 <B>问题</B>：数据集中包含了一个含糖量(sugars=1)为1的谷物 cheerios(数据概览中，黑色框部分)，其营养价值是50.765,而非估计值的56.98.<br>二者之差也即<B>y-y~</B>=56.98-50.765=6.215,称为预测误差(prediction error)、估计误差(estimation error)或者误差残留(residual error)。<br>为寻求这种预测误差总体尽可能小，最小二乘回归法会选择一条唯一的回归线，满足使得数据集的整体残差平方和达到最小值。有多重方法可以选择，如中位数回归方法，但最小二乘法回归是最常见的。   
<br>

##三 误差评估

###1 最小二乘法估计   

公式如下：   
![公式](/images/blog/regression6.png)<br>  
<font color="blue">其中误差项e引入用以解释不确定性的因素。</font><br>
<B>基本假设</B><br>
<ul><li>(1)零均值假设：误差项是期望为零的随机变量，即E(e)=0</li><li>(2)不变方差假设：误差项e的方差（用σ2表示）是常数且与x1,x2,....的值无关</li><li> (3)独立性假设：e的变量是相互独立的</li><li>(4)正态性假设：误差项e是正态随机变量</li></ul>
<B>也即：误差项e的值是独立的正态分布随机变量，带有均值0和不变方差σ2</B><br>
回应变量y的分布:   
<ul><li>(1)根据零假设，回应变量y的值均落在回归线上</li><li> (2)根据不变方差假设，不论预测变量x1,x2,..取什么值，y的方差不变</li><li> (3)根据独立性假设，对任意的x1,x2,..取值，y的值都是相互独立的</li><li>(4)根据正态性假设，回应变量y也是正态分布的随机变量。</li></ul>
<B>也即回应变量y也是独立正态变量，均值不变，方差不变。</B><br>
最小二乘回归线(least-square line)将误差的平方和最小化，总的预测误差用SSEp表示，则总的误差平方和为：   
![预测误差](/images/blog/regression7.png)<br>  
利用微积分，在以下微积分方程结果为0的时候,b0和b1的取值会让总的误差平方和最小。关于b0和b1的偏微分方程为:<br>  
![预测误差](/images/blog/regression8.png)<br>   
令上式为0，则有:<br>
![预测误差](/images/blog/regression9.png) <br>
分别求和，得到：<br>
![预测误差](/images/blog/regression10.png)<br>
重新表示为：<br>
![预测误差](/images/blog/regression11.png)<br>  
求出b0和b1的值：<br>
![预测误差](/images/blog/regression12.png)<br>

###2 决定系数

r**2称为决定系数（coefficient of determination）用来衡量回归线的拟合度，也即最小二乘回归线产生的线性估计与实际观测数据的拟合程度。前面提到y^代表回应变量的估计值，y-y^代表预测误差或残差。<br>

####引子

<ul><li>a.想象一下，如果不考虑数据集中含糖量而直接预测其营养价值，我们直观的做法是求其平均值作为预测值。假设开始为数据集里的每个记录计算(y-y')（其中y'为回应变量的平均值），然后计算其平方和，这与计算误差(y-y^)，然后计算误差平方和类似。这时统计量总体误差平方和SST为：<br><img src="/images/blog/regression13.png"> ![预测误差](/images/blog/regression13.png)<br> SST，也称为总体平方和(sum of squares total，SST)是在没有考虑预测变量的情况下，衡量回应变量总体变异的统计量
</li>b.接下来是衡量估计回归方程能多大程度提高估计的准确度。
    运用回归线时的估计误差为：y-y^ ,当忽略含糖量信息时，估计误差是y-y'。因此改进量是：y^-y'.
    进一步基于 y^-y'构造一个平方和的统计量，这样的统计量被称为回归平方和(sum of squares of regression,SSR)，是相对于忽略预测信息，衡量在使用回归线后预测精度提高的统计量，即：<br>![预测误差](/images/blog/regression14.png)<br>由:   y-y'=(y^-y')+(y-y^)  两边都进行平方，然后进行总和运算，有：![预测误差](/images/blog/regression15.png)<br><li>
  </li></ul>

####结论   

SST衡量了回应变量变异的一部分，这部分是被回应变量和预测变量之间的线性关系所解释的。然而不是所有的数据点都正好落在回归线上，这意味着还有一部分y变量的变异不能被回归线所解释。SSE可以被认为是衡量不能被x和y之间的回归线所解释的其他变异，包括随机变异。<br>
决定系数r^2，它衡量了用回归线来描述预测变量和回应变量之间线性关系的符合程度<br>![预测误差](/images/blog/regression16.png)<br>













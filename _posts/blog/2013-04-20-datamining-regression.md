---
layout:     post
title:      数据挖掘方法之二：回归模型（简单线性回归）
category: blog
description: 数据挖掘专栏
---
##注：文中所使用代码为R
#一 概念
简单线性回归模型是用于估计一个连续预测变量和一个连续回应变量的线性关系。回归方程或估计回归方程(estimated regression equation,ERE)：   
  y~=b0+b1*x
  其中
      <ul>
         <li>y~是回应变量的估计值</li>
         <li>b0是回归线在y轴上的截距</li>
         <li>b1是回归线的斜率</li>
         <li>b0和b1称为回归系数</li>
      </ul>
          
#二 实例
##数据来源: [谷物数据集](http://lib.stat.cmu.edu/DASL/Datafiles/Cereals.html)
##数据描述：谷物数据集,包含了77种早餐谷物的16个属性对应的营养信息##
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

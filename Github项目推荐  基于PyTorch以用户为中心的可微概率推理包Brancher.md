## Github项目推荐 | 基于PyTorch以用户为中心的可微概率推理包Brancher

AI研习社 [AI研习社](javascript:void(0);) *5天前*

![img](https://mmbiz.qpic.cn/mmbiz_jpg/bicdMLzImlibRiboYcgtAAFwZvvLPUlRkFmiaQ8aCfWBsYib2ic7uVBLAHBtL8m8gYWxDLRdVWaAoASYXjjYclph6NlQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**A user-centered Python package for differentiable probabilistic inference.**

Brancher是一个以用户为中心的Python包，用于可区分的概率推理。

**Site：****https://brancher.org/**

**Github项目地址：**

**https://github.com/AI-DI/Brancher**

Brancher允许使用随机变分推理来设计和训练可微分贝叶斯模型。 Brancher基于深度学习框架PyTorch。

## **Brancher的特点：**

灵活：易于扩展的建模框架，GPU加速的PyTorch后端

集成：易于使用的现代工具箱，支持Pandas和Seaborn

直观：易于学习具有类似数学语法的符号界面

## **入门教程**

通过以下教程在Google Colab中学习Brancher（更多内容即将推出！）

![1560761443947108.png](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

## **安装**

安装PyTorch后，可以从PyPI安装Brancher：

- 

```
pip install brancher
```

或者直接克隆github项目：https://github.com/AI-DI/Brancher

## **建立概率模型**

概率模型是象征性定义的。 随机变量可以创建如下：



- 
- 

```
a = NormalVariable(loc = 0., scale = 1., name = 'a')b = NormalVariable(loc = 0., scale = 1., name = 'b')
```



可以使用算术和数学函数将随机变量链接在一起：



- 
- 
- 

```
c = NormalVariable(loc = a**2 + BF.sin(b),                    scale = BF.exp(b),                    name = 'a')
```



通过这种方式，可以创建任意复杂的概率模型。 也可以使用PyTorch的所有深度学习工具来定义具有深度神经网络的概率模型。

## **示例：****自回归建模**

### **概率模型**

概率模型是象征性地定义的：

```

```

- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 

```
T = 20driving_noise = 1.measure_noise = 0.3x0 = NormalVariable(0., driving_noise, 'x0')y0 = NormalVariable(x0, measure_noise, 'x0')b = LogitNormalVariable(0.5, 1., 'b')
x = [x0]y = [y0]x_names = ["x0"]y_names = ["y0"]for t in range(1,T):    x_names.append("x{}".format(t))    y_names.append("y{}".format(t))    x.append(NormalVariable(b*x[t-1], driving_noise, x_names[t]))    y.append(NormalVariable(x[t], measure_noise, y_names[t]))AR_model = ProbabilisticModel(x + y)
```

### **观察数据** 

一旦定义了概率模型，我们就可以决定观察哪个变量：

- 

```
[yt.observe(data[yt][:, 0, :]) for yt in y]
```

### **自回归变分分布**

变分分布可以是任意结构：



- 
- 
- 
- 
- 
- 
- 
- 

```
Qb = LogitNormalVariable(0.5, 0.5, "b", learnable=True)logit_b_post = DeterministicVariable(0., 'logit_b_post', learnable=True)Qx = [NormalVariable(0., 1., 'x0', learnable=True)]Qx_mean = [DeterministicVariable(0., 'x0_mean', learnable=True)]for t in range(1, T):    Qx_mean.append(DeterministicVariable(0., x_names[t] + "_mean", learnable=True))    Qx.append(NormalVariable(BF.sigmoid(logit_b_post)*Qx[t-1] + Qx_mean[t], 1., x_names[t], learnable=True))variational_posterior = ProbabilisticModel([Qb] + Qx)model.set_posterior_model(variational_posterior)
```



### **推理**

现在模型被激活了，我们可以使用随机梯度下降来执行近似推断：



- 
- 
- 
- 
- 

```
inference.perform_inference(AR_model,                             number_iterations=500,                            number_samples=300,                            optimizer="SGD",                            lr=0.001)
```





![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibTwZibDGwbc506Utic6M0ENDXuRib8vsl24HUiccK5JcxV9uiba1HQRib1Q8LSxlCGXtzWKrb5GIwqlEMow/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**你可能还想看**

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibSNQfWtl36V3oYQPf7T3UkrevE8m2vs67qWibL60lj5Aq7vdbpCQmtmO1axovib59n18khvXhgxibDHg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibSF5RSIA4QLmWoq0zgibxrJmYoRbH2X4rMx1J9O1SAg5brnichddyuoTSOhsbsfTlXPvJVJ8wEyAzxw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**![img](https://mmbiz.qpic.cn/mmbiz_gif/bicdMLzImlibRAS3Tao2nfeJk00qqxX3axIgPV3yia4NPESGdUJEM9vsfw1O4Dg1iat7lVNAmbCMY65ia2pzfBXm5kg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1) 点这里查看本篇更多相关内容**

[阅读原文](https://mp.weixin.qq.com/s?__biz=MjM5ODU3OTIyOA==&mid=2650677438&idx=2&sn=5d70def612e4d42fc1835cca39071d8d&chksm=bec21dcd89b594dbee3e48482d5cfe3e01f83539af664eca8bb57333700c6f14ecb7e22dbc05&scene=0&xtrack=1&key=06b6f34db6d09e01116d534a00a7d753795a90d06a9ecc3d52beb26d82e6042593de50ff28a25cbb009b6c0f9d2cdfb119ed4a593e91aaf7925114d7eedfa483164877cf0c8228eca33a5963ca037cff&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=iqn5fxyAYAEcbOWN8K0hTmIdnQAEbGoAMytUHUJn7mS3BliHEI0JRQI4B417Pox7##)







微信扫一扫
关注该公众号
# Softmaxloss [参考](https://www.zhihu.com/search?type=content&q=softmax%E6%8E%A8%E5%AF%BC)

## 符号定义

$Softmaxloss = crossEntropy(Softmax(z_{i}))$  

设 $x_{i}, z_{i}, q_{i}$ 分别是原始输入、神经元加权、经过Softmax归一化

- $z = wx$
- $q = softmaxz$
- $Loss = CE(q)$

## 求导 - 链式法则

$$\frac{\partial Loss}{\partial w} = \frac{\partial Loss}{\partial q} \cdot \frac{\partial q}{\partial z}\cdot \frac{\partial z}{\partial w}$$

$\frac{\partial z}{\partial w}$ 已知，重点是求前面两项$\frac{\partial Loss}{\partial q} \cdot \frac{\partial q}{\partial z}$

- $\frac{\partial Loss}{\partial q}$ 是交叉熵函数求导  
交叉熵只有$p_{k}=1$才有损失累加，所以只针对$q_{k}$

$$
\frac{\partial Loss}{\partial q} = \frac{\partial Loss}{\partial q_{k}} = -\frac{1}{q_{k}}
$$

- $\frac{\partial q_{k}}{\partial z_{i}}$ 是Softmax函数求导，需要**分类讨论**
  - $i = k$
    $$
    \frac{\partial q_{k}}{\partial z_{k}} = \frac{e^{z_{k}}\sum - e^{z_{k}}e^{z_{k}}}{\sum^2} = q_{k} - q_{k}^2
    $$
  
  - $i \neq k$
    $$
    \frac{\partial q_{k}}{\partial z_{i}} = \frac{0\sum - e^{z_{k}}e^{z_{i}}}{\sum^2} = q_{k} q_{i}
    $$

结合两个导数
$$
\frac{\partial Loss}{\partial z_{i}} = q_{i} - p_{i}
$$

所以当用softmaxloss作为损失函数，梯度传播是当标签是1，预测值减去1；当标签是0，不变

**数值稳定**  
为了稳定输出，要统一减去最大的值 $D$
$$
D = max(z_{1},z_{2}....z_{n})
$$

## Softmax 和 多个 Logistic 区别

普通的 Logstic 回归是二分类问题，而 Softmax 是多分类问题，要想实现多分类，需要改进 Logistic 回归，具体做法是：

- 每个类别都建立一个二分类器，属于该类别的为1，其他为０
- 对于选择softmax还是k个logistics回归，取决于所有类别之间是否互斥。
  - 所有类别之间明显互斥用Softmax；
  - 所有类别之间不互斥有交叉的情况下最好用k个logistics分类器。

## 为什么分类问题用交叉熵而不是 MSE

- 交叉熵是凸函数，MSE 是非凸函数
- 交叉熵梯度传播友好，MSE 有梯度消失风险
- 交叉熵只针对正样本，MSE不仅针对正样本，还要平均负样本，这在分类问题中没有必要

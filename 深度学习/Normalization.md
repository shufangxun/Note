# Normalization

![1575014020(1).png](https://i.loli.net/2019/11/29/agmjAWbVryOEvJo.png)

**一句话总结:** 减去均值，除以标准差，再施以线性映射，缓解梯度消失问题
$$
y = \gamma \cdot (\frac{x - \mu(x)}{\sigma(x)}) + \beta
$$

- BN: 批样本单通道
- LN: 单样本全通道  
- IN: 单样本单通道  
- GN：单样本批通道

## Batch Normalization

![1575014377(1).png](https://i.loli.net/2019/11/29/rEADuU9sZlIgife.png)

训练

- 每个 mini batch 上求均值，在N、H、W上操作，保留**通道 C 的维度**，滑动平均保存参数
- 每个通道 C 学习仿射变换 $\gamma$ 和 $\beta$

测试

- 直接用保存的参数做处理

#### 优点

**1. 可以使用大的学习率加速训练**  
对输入层的输入特征进行归一化，从而改变损失函数的形状，使得每一次梯度下降都可以更快的接近函数最小值点，从而加速模型训练过程

**2. 减轻了对参数初始化的依赖，利于调参**  
Batch Norm削弱了前层与后层之间的联系，使得网络的每层都可以自己进行学习，相对其他层有一定的独立性

**3. 正则化效果**
每次计算 $\mu$ 和 $\sigma$ 都是在一个 mini-batch 上进行计算，而不是在整个数据样集上，带来一些比较小的噪声

#### 缺点

**1. 性能依赖于 batch size**
一般来说每GPU上batch设为32最合适
- 过小的batch size会导致其性能下降；
- 过大的batch size会超过内存容量，需要跑更多的epoch，导致总训练时间变长；会直接固定梯度下降的方向，导致很难更新

**2. 测试与训练的 $\mu$ 和 $\sigma$ 有偏差**
训练的时候在训练集上通过**滑动平均**预先计算好$\mu$ 和 $\sigma$，测试的时候直接调用这些预计算好的来用，但是，当训练数据和测试数据分布有差别时，在训练，验证，测试这三个阶段存在 inconsistency

#### 卷积层和Batch层融合加速

![1575016451(1).png](https://i.loli.net/2019/11/29/CstbeXFJQZw85io.png)

## Group Normalization

训练

- 独立于 Batch，将通道 C 分为 G 组，每组计算 $\mu$ 和 $\sigma$，组内共享，不滑动平均记录这两个参数
- 每个通道 C 学习仿射变换 $\gamma$ 和 $\beta$

测试
test 也是将通道 C 分为 G 组，每组计算 $\mu$ 和 $\sigma$，组内共享

#### 有效性分析

- 通道间的特征是相关的，传统的方法如 SIFT、HOG等都是按 Group 构建
- 卷积提取的是一种非结构化的特征向量
  - 本源卷积核和经过transform过的变换卷积核，在同一张图像上学习到的特征应该是具有相同的分布，那么，具有相同的特征可以被分到同一个 Group
  - 每一层有很多的卷积核，它们学习到的特征并不完全是独立的，某些特征具有相同的分布，因此可以被 Group

#### 优点

test 也是将通道 C 分为 G 组，每组计算 $\mu$ 和 $\sigma$，组内共享

#### 缺点

正则化效果比 BN 差，可以结合某些正则化方法

#### 代码

```python
def GroupNorm(x, gamma, beta, G, eps=1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    return x * gamma + beta
```

## Layer Normalization

LN 不需要批训练，对每个样本所有层做归一化，不依赖于batch size和输入sequence的长度，因此可以用于batch size为1和RNN中

## Instance Normalization

IN 对每个样本的每一层做归一化，多用在图像风格化中，因为生成结果主要依赖于某个图像实例

## 问答

**1.为什么说有正则化效果？**
   因为训练时每个batch计算均值和方差，引入了不确定性，而测试时是用整体的均值和方差，泛化能力更强

**2.为什么归一化之后还要学习β和γ去变换？**
   Batch Normalization 是为了防止落入饱和区，但是有一点饱和也不是坏处，学习β和γ去控制每个层缩放和平移的程度，保证数据一定特点

**3. 如何测试时加速**
卷积层和 Batch 层融合加速

**4. 参数 $\gamma$ 和 $\beta$ 的数量**
BN GN IN 是 C；LN 是 normalized_shape 矩阵


## Gram 矩阵

![1575012573(1).png](https://i.loli.net/2019/11/29/dwAM3ZiWLy1xzGo.png)

### 作用
Gram matrix 度量**各自维度的特性**以及**不同维度间的关系**

- 对角线元素提供了不同特征图各自的信息，其余元素提供了不同特征图之间的相关信息
- 当同一个维度上面的值相乘的时候，原来小的变得更小，原来大的变得更大，起到突出自己的特点作用

### 计算
Gram 矩阵 是计算 i 通道的 feature map 与 j 通道的 feature map 的内积

- 特征图 $C \cdot H \cdot W $，$C$ 是通道，$H \cdot W$ 是特征图尺寸
- Gram 矩阵 尺寸是 $C \cdot C \cdot 1 $，$Gram(i,j)$ 是 $i$ 通道的 feature map 与 $j$ 通道的 feature map 的内积值

### 与协方差的区别
Gram矩阵和协方差矩阵的差别在于：Gram 矩阵没有白化，也就是没有减去均值，直接使用两向量做内积。

### 代码
主要是 Gram 矩阵的计算和损失函数定义

- Gram 矩阵

    ```python
    class Gram(nn.Module):
        def __init__(self):
            super(Gram, self).__init__()

        def forward(self, input):
            a, b, c, d = input.size()
            feature = input.view(a * b, c * d)
            gram = torch.mm(feature, feature.t())
            gram /= (a * b * c * d)
            return gram
    ```

- style loss

    ```python
    class Style_Loss(nn.Module):
        def __init__(self, target, weight):
            super(Style_Loss, self).__init__()
            self.weight = weight
            self.target = target.detach() * self.weight
            self.gram = Gram()
            self.criterion = nn.MSELoss()

        def forward(self, input):
            G = self.gram(input) * self.weight
            self.loss = self.criterion(G, self.target)
            out = input.clone()
            return out

        def backward(self, retain_variabels=True):
            self.loss.backward(retain_variables=retain_variabels)
            return self.loss
    ```

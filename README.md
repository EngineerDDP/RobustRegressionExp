# RobustRegressionExp
一个用于测试各种形态的MLP（Multi-Layer Perception）的测试框架

## 环境
项目使用VS2019编写，依赖的Python版本为 3.6 <br>
其他依赖项：
* numpy（any version）
* pandas （any version）
## 结构
项目包含一个主要目录和三个文件<br>
文件结构：
* *Input.py*  定义用于产生随机样本输入的样本产生器。
* *robustReg.py* 定义MLP模型的各个组件。
### 标准预测模型接口
代码中使用的所有算法模型均遵循标准预测模型接口。<br>
```
  class Model:
    def fit_predict(self, x):
      """
        给定一个输入序列 x ，拟合给定的目标分布，并输出预测结果（无监督学习）。
      """
    def fit(self, x, y):
      """
        给定一个输入序列 x 和对应的预测样本标签（label） y ，拟合目标分布。
      """
    def predict(self, x):
      """
        给定一个输入序列 x ，输出其预测结果。
      """
```
### 样本产生器 *Input.py*
```
  class Normalization
```
归一化算法模型，用于对样本数据的归一化，模型有两个方法。<br>
* *fit_predict*方法对训练集进行归一化并学习normalization特征。
* *predict*方法对归一化后的值进行逆操作，获取原值。
构造函数有两个参数：<br>
* 第一个参数*scale*为放缩系数，默认为1.0，几乎用不到非默认值。
* 第二个参数*mode*为归一化方式，参数为字符串，*'z'*代表使用标准分数和（Z-socre）进行归一化，*'m'*代表使用min-max方法进行归一化。
样例：
```
>>> model = Normalization()
>>> train = np.random.uniform(0, 100, size=[10])
>>> train
array([42.94969124, 72.9259261 , 12.96497754,  1.16109273, 70.03028099,
       94.73401369, 77.97657706, 89.29123556, 34.20313184, 35.72549344]) 
>>> result = model.fit_predict(train)
>>> result
array([-0.3339905 ,  0.64309709, -1.31135446, -1.69610689,  0.54871236,
        1.35394059,  0.80772512,  1.17653102, -0.61908817, -0.56946617])
>>> model.predict(result)
array([42.94969124, 72.9259261 , 12.96497754,  1.16109273, 70.03028099,
       94.73401369, 77.97657706, 89.29123556, 34.20313184, 35.72549344])

```
.
```
  class NoiseSimulation
```
噪声产生器算法模型，给指定的输入添加指定分布情况的噪声。模型有一个方法。
* *predict*给定的输入 x ，输出其添加噪声后的结果 x'
构造函数有四个参数：<br>
* 参数 *normal_scale* 代表在产生噪声的时候使用的一维高斯过程的标准差 sigma（σ）
* 参数 *bin_scale* 代表在产生噪声的时候使用的二项分布正负样本间的差的绝对值。
* 参数 *bin_rate* 代表二项分布概率 p。
* 参数 *oneside* 代表是否使用有偏椒盐噪声，为True时椒盐噪声总是偏向一侧，为False时则两侧均有椒盐噪声（此时所有产生测噪声其和趋向于0）。
样例：
```
>>> model = NoiseSimulation(1.0, 10.0, 0.1, True)
>>> sample = np.arange(0, 100, 1)
>>> import matplotlib.pyplot as plt
>>> x = sample
>>> y = x
>>> y_1 = model.predict(y)
>>> plt.plot(x, y, 'g.')
>>> plt.plot(x, y_1, 'r.')
```
结果如下：<br>
![效果图](https://github.com/EngineerDDP/RobustRegressionExp/blob/master/RobustRegressionExp/Figure_1.png)
```
  class LinearSimulation
```
线性产生器算法模型。
* 参数 *w* *b* 为一维线性模型（*y=wx+b*）的参数
* 其余四个参数同噪声产生器算法模型
样例：
```
>>> model = LinearSimulation()
>>> x = np.linspace(0, 1, 20)
>>> y_n = model.predict(x)
>>> y_b = model.baseline(x)
>>> plt.plot(x, y_n, 'r.')
>>> plt.plot(x, y_b, 'g-')
```
结果如下：<br>
![效果图](https://github.com/EngineerDDP/RobustRegressionExp/blob/master/RobustRegressionExp/Figure_2.png)
```
  class SinSimulation
```
正弦波产生器算法模型。
* 参数 *a* *b* *w* 为一维线性模型（*y=asin(2πx/w+b)*）的参数
* 其余四个参数同噪声产生器算法模型
样例：
```
>>> model = SinSimulation()
>>> x = np.linspace(0, 10, 40)
>>> y_n = model.predict(x)
>>> y_b = model.baseline(x)
>>> plt.plot(x, y_n, 'r.')
>>> plt.plot(x, y_b, 'g-')
```
结果如下：<br>
![效果图](https://github.com/EngineerDDP/RobustRegressionExp/blob/master/RobustRegressionExp/Figure_3.png)
### 框架构件 *robustReg.py*
* *class RobustLinearRegressionCPU* 整合式的线性鲁棒回归（Robust Regression）模型，基于梯度下降算法，使用纯CPU计算。
* 激活函数<br>

| 类名 | 介绍 | 原函数 | 导函数 |
|:---|:---|:---|:---|
|*Linear*|线性激活函数|_y=x_|_y=1_|
|*Sigmoid*|S型激活函数|_y=1/(1 + exp(-1 * (x + Δ)))_|_y=y*(1-y)_|
|*Tanh*|双曲正切函数|_y=tanh(x)_|_*y=1-y^2*_|

* 全连接层<br>

| 方法|介绍|
|----|----|
|*logit*|逻辑输出（可以理解为神经元输入）|
|*F*|经过激活函数后的输出|
|*backpropagation*|反向传播输出（输出反向传播梯度）|

* 优化器<br>

|方法|介绍|
|----|----|
|*GradientDecentOptimizer*|梯度下降优化器，根据批次大小不同可以作为Mini-Batch GD、SGD、BGD来使用|
|*AdagradOptimizer*|Adagrad优化器|

* 损失函数<br>

|方法|介绍|
|----|----|
|*MseLoss*|均方误差|
|*CrossEntropyLoss*|交叉熵损失|
|*CrossEntropyLossWithSigmoid*|交叉熵损失，与sigmoid一同计算|
|*TanhLoss*|双曲正切损失|

* 模型框架*Model*，用于调用其他模块进行训练。

## 样例
```
>>> sim = LinearSimulation(b=1.0,w=0.4,normal_scale=0.3,bin_scale=10,oneside=True,bin_rate=0.3)
>>> x = np.linspace(0, 10, 100)
>>> y = sim.predict(x)
>>> x = x.reshape([-1, 1])
>>> y = y.reshape([-1, 1])
```
```
>>> nn = []
>>> nn.append(FCLayer(units=1, act=Linear()))
>>> nn.append(FCLayer(units=1, act=Sigmoid()))
>>> loss = MseLoss()
>>> op = GradientDecentOptimizer(loss=loss, layers=nn, learnrate=0.1)
>>> model = Model(nn, op, onehot=False, debug=True)
>>> model.fit(x, y, epochs=20, batch_size=50, minideltaloss=None)
>>> model.predict(x)
```

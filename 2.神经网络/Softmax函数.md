# `Softmax`函数

## 1.`Softmax`函数

1.在`softmax` 回归和具有`softmax输出层`的神经网络中，都会生成`N`个输出，并选择一个输出作为预测类别。在这两种情况下，向量 $\mathbf{z}$ 都是由线性函数生成的。

2.`softmax`函数将 $\mathbf{z}$ 转换为概率分布，如下所述：

- 应用`softmax`函数后，每个输出都将介于 `[0, 1]`，且输出的和为 `1`，以便它们可以解释为**概率**；
- <img src="C:\Users\chen\AppData\Roaming\Typora\typora-user-images\image-20231222143906546.png" alt="image-20231222143906546"  />

3.`softmax`函数可以写为：
$$
\mathbf{a}_j = \frac{e^{\mathbf{z}_j}}{\sum_{k=1}^N e^{\mathbf{z}_k}}	\tag{1}
$$
输出 $\mathbf{a}$ 是一个长度为 `N` 的向量，因此对于 `softmax`回归，还可以这样编写：
$$
\mathbf{a}(x) = 
\begin{bmatrix} 
P(y=1|\mathbf{x};\mathbf{w},b) \\
\vdots \\
P(y=N|\mathbf{x};\mathbf{w},b) \\
\end{bmatrix}
= \frac{1}{\sum_{k=1}^N e^{\mathbf{z}_k}} 
\begin{bmatrix}
e^{\mathbf{z}_1} \\
\vdots \\
e^{\mathbf{z}_N} \\
\end{bmatrix} \\

\tag{2}
$$
**这表明输出是一个概率向量**，其中，输出的概率向量中的第一个元素 $P(y=1|\mathbf{x};\mathbf{w},b)$  是在给定输入 $\mathbf{x}$ 和 参数为 $\mathbf{w},b$ 的条件下为第一个类别的概率。



4.`softmax`的实现：

```python
def my_softmax(z):
    ez = np.exp(z)
    sm = ez / np.sum(ez)
    return sm
```



## 2. 代价函数（交叉熵损失）

1.与`Softmax`相关的**代价函数**（损失函数）为 **交叉熵损失函数**：
$$
L(\mathbf{A},y) = 
\begin{cases}
-log(a_1), & if \space y=1 \\
\vdots \\
-log(a_N), & if \space y=N \\
\end{cases}

\tag{3}
$$
其中 $y$ 是目标类别，$\mathbf{a}$ 是`softmax`函数的输出，且 $\mathbf{a}$ 中的所有元素均代表概率且和为1。

> `Loss` 是单个样本的损失，而`Cost` 是所有样本的平均损失。

请注意，在上面的 $(3)$ 式中，只有与目标相对应的元素才会造成损失（每个样本的目标类别是固定的，则只有满足 $if \space y=target$ 的那一行才会真正的计算损失）。



2.由于`Loss` 只是单个样本的损失，为了写出`Cost`，则我们需要定义一个 ”**指标函数**“，当指数与目标匹配时，该函数将为1，否则为0：’
$$
1\{y==n\}==
\begin{cases}
1, & if \space y==n \\
0, & otherwise
\end{cases}
$$
则代价函数`Cost`为：、
$$
J(\mathbf{w},b) = - \left[\sum_{i=1}^m \sum_{j=1}^N 1\{y^{(i)}==j\} log(\frac{e^{z_{j}^{(i)}}}{\sum_{k=1}^N e^{z_k^{(i)}}}]) \right]  \tag{4}
$$
其中 $m$ 是样本数量，$N$ 是输出的单元数量，$J(\mathbf{w},b)$ 是所有损失的平均值。





## 3.`Tensorflow`中实现 `sofrmax`交叉熵损失的两种方法

### 3.1 最直接的方法

**使用`softmax` 作为最终的全连接层的激活函数**来实现，在`compile` 指令中单独制定损失函数：

```python
model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)
```

- 指定的损失函数为：`SparseCategoricalCrossentropy`
- 在这种实现方式中，`softmax` 发生在最后一层（输出层），损失函数接收`softmax` 函数的输出（一个概率向量）
- **注意**：由于 `softmax`作为激活函数集成到了输出层中，因此输出的是一个 **概率向量**





### 3.2 首选方法（更优的方法、数值稳定的方法）

<img src="C:\Project\MachineLearning\ML\Advanced Learning Algorithms\week2\work\images\C2_W2_softmax_accurate.png" alt="C2_W2_softmax_accurate" style="zoom:75%;" />



如果在训练过程中**将 `softmax` 和 损失函数 结合起来**，可以获得更稳定和准确的结果。

**在这种实现方式中，最后一层（即输出层）选用 linear 作为激活函数（而不是`softmax`）**。由于历史原因，这种形式的输出称为对数。然后在损失函数中有一个附加参数：`from_logits = True`，这告诉损失函数应该将`softmax`操作包含在损失计算中，这将允许优化实数。

```python
preferred_model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')   #<-- Note
    ]
)
preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train,y_train,
    epochs=10
)
        
```



- 在这种实现方式中，输出并不是概率，而是 **[-∞,+∞]** 范围内的实数；
- 在使用这种实现方式的模型进行预测时，必须将输出值输入到`softmax`函数中才能得到每个类别的预测概率；



### 3.3 `SparseCategorialCrossentropy` or ` CategoricalCrossEntropy`

`Tensorflow` 有两种潜在的目标值格式，损失函数的选择定义了预期的格式：

- `SparseCategorialCrossentropy`：期望目标值为与索引对应的整数。例如，如果有10个潜在目标值，则 $y$ 将介于 [0,9] 之间
- `CategorialCrossEntropy`：期望样本的目标值为**独热编码**，其中目标索引处的值为1，而其他 `N-1` 个索引处的值为0。例如，具有10个潜在目标值的样本的目标值（当样本的目标值为2时）为： [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]





## 4. 数值稳定性

### 4.1 `Softmax` 函数的数值稳定性

1.`softmax` 函数的输入是 线性函数 $z_j = \mathbf{w}_j · \mathbf{x}^{(i)} + b$ 的输出，这些数字可能很大，而`softmax`算法的第一步是计算 $e^{z_j}$ ，如果$z_j$  的值过大，这可能会导致**数值溢出错误**。例如运行以下代码将会得到：

```python
for z in [500,600,700,800]:
    ez = np.exp(z)
    zs = "{" + f"{z}" + "}"
    print(f"e^{zs} = {ez:0.2e}")
```

![image-20231222154538656](C:\Users\chen\AppData\Roaming\Typora\typora-user-images\image-20231222154538656.png)

可以看到，如果**指数过大**，则计算 $e^{z_j}$ 的操作将会产生**数值溢出**。



2.为了提高数值稳定性，可以**通过减小指数的大小** 来实现：

- 注意到有：
  $$
  e^{a+b} = e^a e^b
  $$

- 则如果 $b$ 的符号与$a$ 相反，则可以达到减小指数的效果，具体来说，在`softmax`函数中，可以将`softmax`函数上下同时乘以 $e^{-b}$，则有：

  - $$
    a_j = \frac{e^{z_j}}{\sum_{i=1}^{N} e^{z_i}} \frac{e^{-b}}{e^{-b}}
    $$

  - 就可以达到减小指数的效果，而且可以保证`softmax`的输出值不会改变；

  - 且如果 $b$ 是所有 $z_j$ 中的最大值$max(z_j)$，则可以将指数减小到其最小值；

  - 习惯上可以有 $C = max(z_j)$，因为方程对于任何常数 $C$ 都是正确的：

    - $$
      a_j = \frac{e^{z_j - C}}{\sum_{i=1}^{N} e^{z_i-C}} \space where \space C =max(z_j) \tag{5}
      $$

    - ![image-20231222160358975](C:\Users\chen\AppData\Roaming\Typora\typora-user-images\image-20231222160358975.png)



将上述思想通过代码实现：

```python
def my_softmax_ns(z):
    """
    numerically stablility improved
    """
    bigz = np.max(z)
    ez = np.exp(z - bigz)		# minimize exponent
    sm = ez / np.sum(ez)
    return (sm)
```

使用上述实现的版本重新实验：

```python
z_tmp = np.array([500.,600,700,800])
print(tf.nn.softmax(z_tmp).numpy(), "\n", my_softmax_ns(z_tmp))
```

![image-20231222160722844](C:\Users\chen\AppData\Roaming\Typora\typora-user-images\image-20231222160722844.png)



### 4.2 交叉熵损失函数数值稳定性

![image-20231222160811092](C:\Users\chen\AppData\Roaming\Typora\typora-user-images\image-20231222160811092.png)

其中 $a_2$ 是`softmax`函数的输出值，将其展开为：
$$
L(\mathbf{z}) = -log(\frac{e^{z_2}}{\sum_{i=1}^{N} e^{z_i}})	\tag{6}
$$
这是可以优化的，但是**要进行这些优化，必须同时计算`softmax`和损失，而不能先计算`softmax`，然后将`softmax`的输出作为损失函数的输入**（如上述的 [首选实现方法](# 3.2 首选方法（更优的方法、数值稳定的方法） )中所述）。

注意到有：
$$
log(\frac{a}{b}) = log(a) - log(b)
$$
则：
$$
L(\mathbf{z}) = -log(\frac{e^{z_2}}{\sum_{i=1}^{N} e^{z_i}}) = -\left[ log(e^{z_2}) - log(\sum_{i=1}^{N} e^{z_i}) \right]	\tag{7}
$$
其中，通过利用 $ log = ln$ 的等价，可以将其中的第一项简化为 $z_2$，则有：
$$
L(\mathbf{z}) = -\left[ log(e^{z_2}) - log(\sum_{i=1}^{N} e^{z_i}) \right] 
=  - \left[z_2 - log(\sum_{i=1}^{N} e^{z_i}) \right] 
=  log(\sum_{i=1}^{N} e^{z_i}) - z_2	\tag{8}
$$
事实证明，上式中的 $log(\sum_{i=1}^{N} e^{z_i})$ 项会被经常使用，因此有许多库对其都会有一个实现：

- 在 `Tensorflow`中，这是`tf.math.reduce._logsumexp()`

但在该项中，有一个问题是：如果`z_i`太大，还是可能会出现因为指数过大而出现数值溢出的问题。为了解决这个问题，可以借鉴 [`softmax`数值函数的数值稳定性](# 4.1 `Softmax` 函数的数值稳定性) 中减小指数的方法：
$$
log(\sum_{i=1}^{N} e^{z_i}) = log(\sum_{i=1}^N e^{z_i-max(z_j)+max(z_j)}) \\
= log(\sum_{i=1}^{N} e^{z_i - max(z_j)}e^{max(z_j)}) \\
= max(z_j) + log(\sum_{i=1}^{N} e^{z_i-max(z_j)}) \\

\tag{9}
$$
通过这种方法，指数将会被减小从而避免数值溢出的错误产生。

习惯上说 $C = max(z_j)$，因为方程对于任何常数 $C$ 都是正确的，将上述方法实施到损失函数中可得：
$$
L(\mathbf{z}) = C + log(\sum_{i=1}^{N} e^{z_i-C}) - z_2 \space where \space C = max(z_j) 	\tag{10}
$$

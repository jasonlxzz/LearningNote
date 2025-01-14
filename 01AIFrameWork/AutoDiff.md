# 自动微分

[Automatic Differentiation
in Machine Learning: a Survey](https://arxiv.org/pdf/1502.05767)

[Tangent: Automatic Differentiation Using Source
Code Transformation in Python](https://arxiv.org/pdf/1711.02712)

## 符号微分

通过求导法则将指定表达式变换规则，优势：精确数值结果。缺点：表达式膨胀

求导法则1：两个函数相加的导数

$\frac{d}{dx}(f(x)+g(x))-> \frac{d}{dx}f(x) + \frac{d}{dx}g(x)$

求导法则2：两个函数相乘的导数

$\frac{d}{dx}(f(x)*g(x))-> (\frac{d}{dx}f(x))g(x) + (\frac{d}{dx}g(x))f(x)$

## 数值微分

使用有限差分进行近似，优势：容易实现，缺点：计算结果不精确（截断误差，舍入误差），计算复杂度高，

## 自动微分

- 所有数值计算都由有限的基本运算组成

- 基本运算的导数表达式是已知的

- 通过链式法则将数值计算各部分组合成整体

表达式追踪（Evaluation Trace）: 追踪数值计算过程的中间变量

优点：数值精度高，无表达式膨胀。缺点：需要存储中间求导结果，占用大量计算机内存。

### 自动微分模式

- 前向模式
  
  从输入到输出，根据链式求导法则，求出输出对于输入的偏导

- 后向模式
  
  从输出到输入，根据链式求导法则，求出输出对于输入的偏导

- 雅可比矩阵
  
  向量y对于向量x的梯度。

### 实现方式

- 基本表达式（LIB）：1）封装基本的表达式及其微分表达式作为库函数 2）运行时记录基本表达式和相应的组合关系 3）链式法则对基本表达式的微分结果进行组合

- 操作符重载（OO）：1）利用语言多态特性，使用操作符重载基本运算表达式 2）运行时记录基本表达式和相应的组合关系 3）链式法则对基本表达式的微分结果进行组合

- 源码转换法（AST）：1）语言预处理器、编译器和解释器的扩展 2）对程序表达进行分析得到基本表达式的组合关系 3）链式法则对基本表达式的微分结果进行组合

操作符重载的基本流程（pytorch）:

- 操作符重载：预定义特定的数据结构，并对该数据结构重载相应的基本运算操作符

- Tape记录：程序在实际执行时会将相应表达式的操作类型和输入输出信息记录至特殊数据结构

- 遍历微分：得到特殊数据结构后，将对数据结果进行遍历，并对其中记录的基本运算操作进行微分

- 链式组合：把结果通过链式法则进行组合，完成自动微分

操作符重载法的优点：

- 实现简单，只要求语言提供多态的特性能力

- 易用性高，重载操作符后跟使用原生语言的编程方式类似

操作符重载法的缺点：

- 需要显式的构造特殊数据结构和对特殊数据结构进行大量读写，遍历操作，这些额外数据结构和操作的引入不利于高阶微分的实现。

- 对于类似if, while控制流表达式，难以通过操作符重载进行微分规则定义，对于这些操作会退化成基本表达式方法中特定函数封装的方式，难以使用语言原生的控制流表达式

[pytorch自动微分官方教程](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

[pytorch官方case](https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC)

[ascend教程case](https://github.com/chenzomi12/AISystem/blob/main/05Framework/02AutoDiff/06ReversedMode.ipynb)

## pytorch 实现原理

1）反向函数是如何初始化的，

2）核函数是否存在临时内存，device信息怎么存储，内存怎么管理

3）

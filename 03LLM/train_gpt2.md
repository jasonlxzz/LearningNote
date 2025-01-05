# 前言

从[GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA](https://github.com/karpathy/llm.c)开源项目出发学习，关注以下两个方面的知识，进行总结理解，加深记忆。

1. 理清大语言模型预训练的整体流程
   
   - 模型结构梳理
     
     - gpt/gpt2/gpt3
   
   - 训练、验证、测试
   
   - 数据集预处理/后处理，包括tokenizer以及sampling

2. cuda架构以及训练技巧的原理和加速实现
   
   - cuda 架构，api，性能评价
   
   - flash attention
   
   - dp/tp/sp/pipeline 
   
   - DPO
   
   - SFT
   
   - ZeRO-1

# llm.c 项目简介

llm.c是一个简单纯净的C和CUDA实现的大规模语言模型（LLM），无需依赖庞大的PyTorch（245MB）或CPython（107MB）。当前的焦点在于预训练阶段，特别是复现GPT-2和GPT-3的微小系列。

项目目录总览如下：

```bash
root@iZbp16byi4f3fvh4g6jktbZ:~/llm.c# tree -L 1
.
├── dev              #等同于src源码实现，包含cpu/cuda等加速实现
├── doc              
├── LICENSE
├── llmc             #等同于include, 项目头文件
├── Makefile         #编译设置
├── profile_gpt2.cu  
├── profile_gpt2cu.py
├── README.md
├── requirements.txt
├── scripts
├── test_gpt2.c
├── test_gpt2.cu
├── test_gpt2_fp32.cu
├── train_gpt2.c
├── train_gpt2.cu       
├── train_gpt2_fp32.cu
├── train_gpt2.py
└── train_llama3.py
```

最外层的`.c/.cu/.py`可看做sample示例，以下重点解读`train_gpt2.cu`实现细节，作者认为这是目前的最佳实践。

# 大语言模型预训练

## gpt系列

### gpt

[论文地址](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### gpt2

[论文地址](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [开源代码](https://github.com/openai/gpt-2)

#### 研究背景

- **问题定义**：虽然机器学习系统在利用大数据集，高容量模型和监督学习后，在一些任务上表现优秀，但是这些系统较为脆弱，对数据分布和任务的细微变化非常敏感，缺乏泛化能力。目前创建机器学习系统的主要方法是收集一个训练样本数据集，这些样本展示了期望任务中的正确行为，训练系统模仿这些行为，然后在独立同分布（IID）的保留样本上测试其性能。缺陷在于像caption model、reading comprehension systems和image classifiers在面对多样化和多种可能的输入时，常常表现出不稳定的行为。

- **研究意义**：希望朝着通用的系统发展，这些系统能够执行许多任务，最终无需为每个任务手动创建和标注训练数据集。

- **现有方法**：多任务学习Multitask learning，当前的机器学习系统需要数百到数千个样本来归纳出泛化良好的函数。这表明，多任务训练可能同样需要同样多的有效训练对来实现其潜力。继续扩展数据集的创建和修改设计训练目标，以达到当前技术所需的程度，将非常困难。这促使我们探索其他多任务学习的设置。
  
  当前在语言任务上表现最好的系统使用了预训练和监督微调的组合。

#### 方法与技术

- **核心方法**：证明了语言模型可以在零样本设置下执行下游任务——无需任何参数或架构修改。我们通过强调语言模型在零样本设置下执行广泛任务的能力，展示了这种方法的潜力。根据任务的不同，作者取得了有希望的、有竞争力的和最先进的结果。

- **创新点**：大规模语言模型（如 GPT 系列）在零样本学习中的潜力。通过学习互联网上的自然语言数据，模型可以推断并执行多种任务，而无需明确的任务特定训练，展示了无监督多任务学习的可能性。

- **技术细节**：方法的具体实现细节（如算法、模型架构等）。
  
  - 首先构建了WebText数据集，该WebText 的初步版本，未包含 2017 年 12 月之后创建的链接，并且在去重和一些基于启发式的清理后，包含略超过 800 万篇文档，总计 40 GB 的文本。
  
  - 然后建立输入表达，BPE方法(Byte Pair Encoding)，是字符级和词级语言建模之间的一个实用折中方案，BPE 将频繁出现的符号序列视为词级单元，而将不频繁出现的符号序列视为字符级单元。
    
    **字节级 BPE**：基础词汇表仅包含 256 个字节，适合处理多语言文本。
    
    **跨字符类别合并**：禁止 BPE 合并不同类别的字符（如字母和标点符号），减少不合理合并。
    
    **空格处理**：将空格作为特殊字符处理，提高分词效率。
  
  - 模型使用Transformer based架构，细节基本和GPT一代相同，改动点：
    
    1）Layer normalization被加在每个子block之前，一个额外的layer normaliztion加在了最后的自注意力模块后面。
    
    2）使用了一种改进的初始化方法，考虑了随着模型深度增加的残差路径累积。将残差层的权重在初始化时按 1/N 的比例缩放，其中 N 是残差层的数量。
    
    3）词汇表扩展到 50,257。将上下文大小从 512 增加到 1024 个token，并使用更大的batchsize 512。

#### 实验与结果

- **实验设置**：实验数据集、评价指标、对比方法等。
  
  - 实验数据集： WebText（Reddit上抓取下高质量的外链内容，并经过去重，清理和筛选后制作的）
  
  - 评价指标
  
  - 对比方法

- **主要结果**：论文的实验结果如何？是否支持其论点？

- **结果分析**：结果的优缺点是什么？是否有潜在问题？

#### 个人心得

- **收获**：从论文中学到了什么？

- **启发**：论文对你的研究或工作有什么启发？

- **疑问**：论文中是否有不理解或存疑的地方？

- **改进建议**：你认为论文的方法或实验可以如何改进？

### gpt3

[论文地址](https://arxiv.org/pdf/2005.14165) 

## 数据集预处理

### 分词算法

#### BPE

#### WordPiece

## 超参设置

[gpt2](###gpt2) 论文有4种模型结构设置

| Parameters | Layers | d_model | num_head |
|:---------- | ------ | ------- | -------- |
| 117M       | 12     | 768     | 12       |
| 345M       | 24     | 1024    | 16       |
| 762M       | 36     | 1280    | 20       |
| 1542M      | 48     | 1600    | 25       |

[gpt3](###gpt3) 官方有8种模型结构设置

| Parameters | Layers | d_model | num_head | head_size | Batch Size | Learning Rate($10^{-4}$) |
| ---------- | ------ | ------- | -------- | --------- | ---------- | ------------------------ |
| 125M       | 12     | 768     | 12       | 64        | 0.5M       | 6.0                      |
| 350M       | 24     | 1024    | 16       | 64        | 0.5M       | 3.0                      |
| 760M       | 24     | 1536    | 16       | 96        | 0.5M       | 2.5                      |
| 1.3B       | 24     | 2048    | 24       | 128       | 1M         | 2.0                      |
| 2.7B       | 32     | 2560    | 32       | 80        | 1M         | 1.6                      |
| 6.7B       | 32     | 4096    | 32       | 128       | 2M         | 1.2                      |
| 13B        | 40     | 5140    | 40       | 128       | 2M         | 1.0                      |
| 175B       | 96     | 12288   | 96       | 128       | 3.2M       | 0.6                      |

其余设置：

| 模型                                     | gpt2  | gpt3  |
| -------------------------------------- | ----- | ----- |
| max_seq_len                            | 1024  | 2048  |
| vocab_size                             | 50257 | 50257 |
| padded_vocab_size(for cuda accelerate) | 50304 | 50304 |

## 模型创建

- 模型权重
  
  VP(padded_vocab_size)，C(d_model)， max_T(max_seq_len), L(Layers)  
  
  下方列出模型结构所需的所有预训练权重[16个]：
  
  - wte：输入embedding层，Vp * C
  
  - wpe:  输入位置编码层， maxT * C
  
  - ln1w：前置layernorm的权重，L * C
  
  - ln1b：前置layernorm的偏置，L * C
  
  - qkvw：Q/K/V 输入线性变换的权重， L * (3 * C) * C
  
  - qkvb：对应的偏置，L * (3 * C) 
  
  - attprojw：sdpa结束后，多个head 拼接在一起，再进行线性变换权重，L * C * C
  
  - attprojb：对应的偏置，L * C
  
  - ln2w：后置layernorm的权重，L * C
  
  - ln2b：后置layernorm的偏置，L * C
  
  - fcw：FFN的第一层 线性变换权重，C->4C, L * (4 * C) * C
  
  - fcb：对应偏置，L * (4 * C)
  
  - fcprojw：FFN的第二层线性变换权重，4C->C，L * C * (4 * C)
  
  - fcprojb：对应偏置，L * C
  
  - lnfw：最后输出的layernorm, C
  
  - lnfb: 对应偏置，C
  
  注：最后输出会有一个embedding，和输入embedding共享权重。

- 模型激活(每层输出，21个，考虑残差连接的内存复用，element-wise)
  
  - encoded, embedding层输出 (B,T,C)
  
  - ln1，pre layernorm的输出 （L, B, T, C），recompute无需申请
  
  - ln1_mean，pre layernorm的均值（L, B, T）
  
  - ln1_rstd,  pre layernorm的方差 （L, B, T）
  
  - atty，注意力施加在V上的结果 (L, B, T, C)
  
  - att，注意力权重 (L, B, NH, T, T) ； 开启cudnn大小为 (L, B, NH, T)
  
  - residual2， 残差相加结果，(L, B, T, C)
  
  - ln2， post layernorm的输出，(L, B, T, C)，recompute无需申请
  
  - ln2_mean，post layernorm的均值，(L, B, T)
  
  - ln2_rstd，post layernorm的方差，(L, B, T)
  
  - fch，FFN第一个线性变换的输出， (L, B, T, 4*C)
  
  - ftch_gelu， FFN第一个线性变换+gelu输出，  (L, B, T, 4*C)，recompute无需申请
  
  - residual3，残差相加结果，(L, B, T, C)
  
  - lnf， 最后的layernorm输出， (B, T, C)
  
  - lnf_mean， 最后的layernorm均值， (B, T)
  
  - lnf_rstd， 最后的layernorm方差， (B, T)
  
  - losses, 损失函数输出 (B, T)
  
  - qkvr， 
  
  - output， 最终输出
  
  - scratch_bt4c，
  
  - scarach_btc，

- 内存分配
  
  - 权重：cudaMalloc一块大内存，然后通过每个权重大小在大内存上进行偏移。
  
  - 梯度：与权重大小相同，偏移方式与权重相同
  
  - 激活activation(每层的输出)：存在复用以及一些特殊buffer，后续补充理解
  
  - 优化器：使用的AdamW，一阶距+二阶距+master weight。三种权重，多卡训练时可以分配到不同的卡上。

## 权重初始化

按照上面列出的，总共有16种类型权重，总共9weight+7bias。bias初始化为0.

- wte/wpe只会初始化一次，服从正态分布$W \sim \mathcal{N}\left(0, 0.02 \right)$

- 所有的layernorm weight 都初始化为1（总共有三个，只会初始化一次）

- wqkv/fcprojw初始化为： $W \sim \mathcal{N}\left(0, 0.02 \right)$

- attnprojw/fcprojw初始化为: $W \sim \mathcal{N}\left(0, 0.02 \cdot \frac{1}{\sqrt{N}}\right)$

这里的N=2L，一个L存在两个残差层，论文里面N就是残差层的个数。

另外正态分布会有一个随机种子，torch的随机种子如下：

```python
# 设置随机种子
torch.manual_seed(42)
```

cpu权重内存初始化后拷贝到gpu权重内存，然后释放cpu权重内存。

## 数据集加载



## 模型训练

## 模型验证测试

# 大模型加速技巧

## 基本结构

### layernorm

常用于NLP，保证每个token的特征保持独立分布。输入N,C,H,W, 在CHW上进行归一化。

**注**：区别于batch norm，常用于图像处理，卷积CNN网络输出的C，是利用不同的滤波器得到。因此某一个通道可以看成是某一种特征，这类特征拥有旋转不变性，在N,H,W上进行归一化。

## Zero Redundancy Optimizer stage



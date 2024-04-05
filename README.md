

<h1 align="center">ACTIVE_LEARNING_PAPER</h1>

<div align="center">
    Contributed by 
    <a href="https://github.com/yhr-code">yhr-code</a>
</div>

****
## Table of Contents
<!-- TOC -->

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Papers](#papers)
    - [主动学习综述](#主动学习综述)
    - [主动学习的缺点/对主动学习的批判](#主动学习的缺点/对主动学习的批判)
    - [CV](#CV)
    - [LLM(NLP)](#LLM(NLP))
    - [L2D(Learning_to_defer)](#L2D(Learning_to_defer))
   


<!-- /TOC -->

****
## Introduction


This is the paper list for active learning in **CV, NLP，L2D(Learning to defer) and LLM**, which pays attention to the sampling strategy, thus enabling the model the get **the most valuable samples** for training by **maintaining the input samples of the model to the greatest extent under the condition of low labeling budget.**

***根据 `https://github.com/SupeRuier/awesome-active-learning` 中 有关cv nlp llm 和 L2D 的内容,加上自己的总结 故整理成一个仓库 希望能把active learning 的思想更好的用在以上领域里面***

本人理解的active learning实际上可以分成两种概念：

* **传统意义上:** active leanring 是一种人机协作的实例，让人类专家去迭代选择样本池(pool)/样本流(stream) 中他们认为最有价值的样本去给机器模型学习，这种观点需要数据样本对应的人类专家。

* **现实实际指：** 由于我们认为人类专家是通过一种直觉，一种缺乏完全理性的判断的选择策略去选择样本；人类专家的状态也可能波动。由此现实实际的active learning实际上是一种能与模型交互的一种算法策略：通过迭代给予模型样本，通过模型的反馈去不断修正样本选择策略的过程（当然，有一些存在的active learning algorithms 没有使用迭代的方法，但是少数；大多数主动学习的方法的核心就是**迭代**这个思想）


**Problem - 面对的问题：** 高标签成本在机器学习社区中很常见。获取大量注释阻碍了机器学习方法的应用。

**Assumption - 假设：** 并非所有实例对于所需任务都同样重要，因此仅标记更重要的实例可能会带来成本降低。

The author's email: `yehaoranchn@gmail.com`

****
## Papers

***由于如今active learning 分类方式复杂，作者针对我们所关注的几个方面把active learning 有关论文按以下几个方面分类：***

* 主动学习综述
* 主动学习的缺点/对主动学习的批判
* CV
* LLM(NLP)
* L2D(Learning_to_defer)


论文都以下列格式为后缀：
- [![](https://img.shields.io/badge/PendingReview-e2fbbe)]()：表示还没读过
- [![](https://img.shields.io/badge/Overviewed-366588)]()：表示泛读
- [![](https://img.shields.io/badge/DetailedReviewed-0c1f2f)]()：表示细读

Eg. `ACL-2023` **Title** [paper] [code] .. [authors][![](https://img.shields.io/badge/PendingReview-e2fbbe)]()

### 主动学习综述

这一部分的论文是从主动学习的综述入手，可以快速了解这一领域的大致情况：

- `University of Wisconsin-Madison Department of Computer Sciences` **Active learning literature survey** [[paper](https://minds.wisconsin.edu/handle/1793/60660)][[code]][Settles, Burr][![](https://img.shields.io/badge/DetailedReviewed-0c1f2f)]()  **优先读这一篇active learning 综述，里面用了较少的篇幅把整个active learning 框架讲的比较明白**
- `Foundations and Trends in Machine Learning, 2013` **Theory of active learning/ A statistical theory of active learning**[[paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=98342bdb6d7de1e0189428d9049d75b001d7d29b)][[code]][Steve Hanneke][![](https://img.shields.io/badge/Overviewed-366588)]() **这一篇active learning 详尽系统总结了active learning发展框架与不同选择策略 内容200多页 而且有关于active learning数理分析的部分**
- `arXiv 20` **A Survey of Deep Active Learning** [[paper](https://arxiv.org/pdf/2009.00236.pdf)][[code]][Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Brij B. Gupta, Xiaojiang Chen, Xin Wang][![](https://img.shields.io/badge/Overviewed-366588)]() **这一篇文章总结了deep learning 时代主动学习研究，并且给出了一些发展方向**
- `arXiv 22` **A Comparative Survey of Deep Active Learning** [[paper](https://arxiv.org/pdf/2203.13450.pdf)][[code]][Xueying Zhan, Qingzhong Wang, Kuan-hao Huang, Haoyi Xiong, Dejing Dou, Antoni B. Chan][![](https://img.shields.io/badge/Overviewed-366588)]() **这一篇文章总结了deep learning 时代主动学习的各项研究，比较详尽，可以跟arXiv20这篇一起看**

### 主动学习的缺点/对主动学习的批判
但是，在deep learning 时代，已经有比较多的paper在抨击主动学习这个方向，知乎上有一个博主讲深度主动学习的方向比较好，现给出链接：

主动学习（Active Learning）近几年的研究有哪些进展，现在有哪些代表性成果？ - 温文的回答 - 知乎 https://www.zhihu.com/question/439453212/answer/2147806195

**原因笼统来说：**
* 比不过竞争对手: semi-supervised learning /self-superivsed learning + fine-tune
* 十分依赖应用(或数据)本身, 泛化能力不强无法提供端到端的解决方案
* 跟其他组件结合的不好(如有一些论文在研究对于图像分类的ssl, 加入主动学习的算法的效果甚至不如 random sampling, 有人认为是主动学习算法对一些sota的ssl算法中的数据增强不适配；强的数据增强会模糊掉主动学习选取的样本的策略)

(但是不要灰心，一些特定领域的active learning 还是比竞品有效果的。)

#### 主动学习会给系统带来偏差
这种偏差是不可避免的:由于主动学习只是在预算中选择样本，所以说这些样本的分布是不能代表整体的样本分布的

- `ICLR 21` **On Statistical Bias In Active Learning: How and When to Fix It** [[paper](https://openreview.net/pdf?id=JiYq3eqTKY)][[code]][Sebastian Farquhar, Yarin Gal
, Tom Rainforth][![](https://img.shields.io/badge/PendingReview-e2fbbe)]() **主动学习不仅可以作为一种减少最初设计的方差的机制，而且还因为它引入了一种可以主动提供帮助的偏差通过正则化模型**
- `WACV 24` **Critical Gap Between Generalization Error and Empirical Error in Active Learning** [[paper](https://openaccess.thecvf.com/content/WACV2024/papers/Kanebako_Critical_Gap_Between_Generalization_Error_and_Empirical_Error_in_Active_WACV_2024_paper.pdf)][[code]][Yusuke Kanebako][![](https://img.shields.io/badge/Overviewed-366588)]() **除了 AL 选择的数据之外，还有大量带注释的数据可用于评估模型性能的假设是不现实的。因此，在使用 AL 构建实际模型时，应仅使用 AL 选择的数据通过交叉验证来估计实际生产环境中的泛化误差。对AL选择的数据进行交叉验证时，实际泛化误差与经验误差之间存在差距**

#### 主动学习冷启动问题
冷启动是指在预算较低的情况下，主动学习算法应该从哪个地方开始是一个需要考虑的问题。一些论文指出在低预算的情况下，随机选择样本的效果优于大多数深度主动学习策略

- `ICML 14` **Cold-start Active Learning with Robust Ordinal Matrix Factorization** [[paper](https://proceedings.mlr.press/v32/houlsby14.pdf)[[code]][Neil Houlsby,Jose Miguel Hernandez-Lobato,Zoubin Ghahramani][![](https://img.shields.io/badge/Overviewed-366588)]() **本文讨论了主动学习的冷启动问题并给出了使用贝叶斯方法解决冷启动问题，对评分数据进行行矩阵分解，利用参数的后验分布跟用户物品之间的不同噪声水平的似然函数去计算不确定性，最后扩展了"BALD"的新的主动学习框架**
- `ICML 22` **Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets** [[paper](https://arxiv.org/pdf/2202.02794.pdf)][[code]][Guy Hacohen,Avihu Dekel,Daphna Weinshall][![](https://img.shields.io/badge/PendingReview-e2fbbe)]() **本文认为在主动学习中冷启动的问题是因为在预算较少的时候使用不确定性采样会使得训练的模型容易过度拟合，并且往往会做出过于自信的预测。依赖这些预测通常会导致嘈杂的不确定性信号。文章发现从“简单”样本开始，逐渐增加“难度”水平是可取的。主动学习中的类似见解是，当预算较低时选择“简单”样本，而当预算较高时选择“困难”样本。 文章提出了一种低预算“简单选取”策略 TypiCLust 使用Kmeans 密度估计去选择样本，而在高预算的时候可以转向对样本不确定性的采样(margin) 具体内容可以见作者的博客：https://avihu111.github.io/Active-Learning/**

#### 一些主动学习算法跟其他组件结合效果有可能不太好
有论文表明，在分类任务和语义分割任务中，SSL-AL的效果可能不如SLL-Random Sampling

- `arXiv 19` **Parting with Illusions about Deep Active Learning** [[paper](Parting with Illusions about Deep Active Learning)][[code]][Sudhanshu Mittal Maxim Tatarchenko Ozg ¨ un C¸ ic¸ek Thomas Brox] [![](https://img.shields.io/badge/DetailedReviewed-0c1f2f)]()：**这项工作指出，当前最先进的 DeepAL 工作没有考虑“半监督学习”、“数据增强”等并行设置。因此，他们对几种 AL 策略与 SL 和 SSL 训练范式进行了比较研究。他们对两个任务进行了实验：图像分类和语义分割。**

    分类任务的结果:
    
    * AL 与数据增强配合得很好，但数据增强模糊了 AL 策略之间的差异：它们的表现基本相同。
    * 结合 SSL 和 AL 可以比原始 SSL 产生改进。
    * AL方法的相对排名在不同数据集上完全变化
    * AL选择策略在低预算情况下会适得其反，甚至比随机抽样更糟糕。
    * 在高预算和低预算设置中，SSL-AL 方法明显优于预训练 ImageNet 网络的微调。
    
    语义分割任务的结果：
    
    * 使用 SSL 进行随机选择效果最佳
    
    总体结论：
    
    * 目前主动学习中使用的评估协议不是最优的，这反过来又导致对方法性能的错误结论。
    * 在传统主动学习环境中应用的现代半监督学习算法显示出更高的相对性能提升。
    * 最先进的主动学习方法通​​常无法优于简单的随机抽样，尤其是当标签预算很小时。


#### 大多数主动学习论文中的策略鲁棒性泛化性不强
有论文发现不同的论文的AL策略需要通过大量的微调形成，而且鲁棒性和泛化性不高

- `CVPR 22` **Towards Robust and Reproducible Active Learning using Neural Networks** [[paper](https://arxiv.org/pdf/2002.09564.pdf)][[code]][Prateek Munjal, Nasir Hayat, Munawar Hayat, Jamshid Sourati, Shadab Khan][![](https://img.shields.io/badge/PendingReview-e2fbbe)]() **这项工作指出，随机抽样基线和 AL 策略的性能在不同论文中存在显着差异。为了提高 AL 方法的再现性和鲁棒性，在这项研究中，他们在公平的实验环境中与随机采样相比，评估了这些图像分类方法的性能。他们还指出，大多数 AL 工作都忽略了正则化，而正则化可以减少泛化误差。他们对不同的正则化设置进行了比较研究。 （参数范数惩罚、随机增强 (RA)、随机加权平均 (SWA) 和抖动 (SS)）**
  
  图像分类任务的结果：
  
  * RS 的性能明显优于他们在其他作品中所说的。而且没有任何策略的表现明显比 RS 更好。
  * 对于不同的 AL 批量大小，策略的性能不一致。
  * AL 方法的性能并不优于 RS，并且它在类不平衡设置上并不稳健。
  * 使用 RA 和 SWA 训练的模型在所有 AL 迭代中始终实现显着的性能提升，并且在多次实验运行中表现出明显较小的方差。
  * 考虑从VGG16到ResNet18和WRN-28-2的选定实例，性能各不相同。 RS的表现还是不错的。


### CV
由于作者倾向于CV中图像分类板块 所以说CV这一类只会出现部分子类



#### 综述
``

#### 图片分类


### LLM(NLP)
在LLM之下的active learning的应用，在近几年也是有比较多的论文关注于在LLM模型的active learning sampling的情况。
因为时间限制，作者先从finetuning的方向来引荐有关active learning的论文:
(请注意，这里的active learning 使用了非常广泛的概念，即只要是一种挑选样本的策略，作者这里都把其归类到active learning的概念里面,即使文章没有提及active learning的概念)

- `ACL 24 在投` **STAR: Constraint LoRA with Dynamic Active Learning for Data-Efficient Fine-Tuning of Large Language Models** [[paper](https://arxiv.org/pdf/2403.01165.pdf)[[code]][Linhai Zhang,Jialong Wu,Deyu Zhou ,Guoqiang Xu]-[![](https://img.shields.io/badge/PendingReview-e2fbbe)]()：**这篇文章认为在PFET和MEFT高效微调形式外，LLMs微调中被忽视的因素之一是任务的固有复杂性，微调LLMs所需的人工标注资源也很重要。所以说DFET(Data Efficient Fine-TUning)基于样本选择的高效微调方式也十分重要，而且是第一篇在LLM推理领域使用传统意义上的active learning方法(需要k次迭代分别选样本)与PFET方法(Lora) 结合达到DEFT的目的。本文指出之前盲目直接结合PEFT+al效果不好（不确定性差距和模型校准不佳）故提出了一种新型的DEFT 方法 STAR(conStrainT LoRA with dynamic Active leaRning)：**

  **总体定义：**
  这是一种有效整合基于不确定性的主动学习和LoRA的新方法改善盲目结合方法的缺陷:
  - 针对**不确定性差距**，作者引入了一种**动态不确定性测量**，它在主动学习迭代过程中结合了基础模型和完整模型的不确定性。
  - 对于**模型校准不佳**，在LoRA训练过程中引入了**混合正则化方法**，以**防止模型过于自信**，并采用**蒙特卡洛dropout机制**来增强不确定性估计。
  
  **总体方法流程：**
    ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/f2655973-edd1-476b-a6a8-e3e5dc7c6f05)
    ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/2d6cf0c7-12da-406c-859b-07253eae3d42)


    1. 模型推断：利用当前模型 $M_k$ 对未标记数据集 $D^U_{k}$ 进行推断。
    2. 数据查询：基于动态不确定性估计方法，选择最具信息量的示例，形成子集 $S^U_k$。
    3. 数据标记：对未标记子集 $S^U_k$ 进行标记，形成标记子集 $S^L_k$ 。
    4. 数据集更新：通过将标记子集追加到已标记数据集 $D^L_k$ 中更新已标记数据集 $D^U_{k+1} = D^U_{k} \cup S^L_{k}$ 。
    5. 模型训练：利用新标记数据集 $D^U_{k}$ 更新当前模型，得到下一次迭代的模型 $M_{k+1}$ 。
 
  **具体算法：**
   - **不确定性测量：**
     最大熵（ME）和预测熵（PE）都是用来评估模型预测的不确定性的指标，但它们在方法和所包含的信息方面有所不同。

        最大熵（ME）以其与黄金响应的独立性为特征。它通过计算所有可能结果的熵，定量评估模型预测的不确定性，公式如下：
     
        $$
\text{ME}(s, x) = - \sum_{i=1}^{N} \sum_{j=1}^{V} p(v_{ij}|s<i, x) \log p(v_{ij}|s<i, x)
$$

        其中，\( s \) 是生成的响应，\( p(v_{ij}|s<i, x) \) 是在 \( s \) 的第 \( i \) 个元素中词汇表的第 \( j \) 个标记的概率，\( V \) 是词汇表的大小。
        预测熵（PE）融入了对黄金响应的依赖性，提供了给定预测分布的真实标签的预期信息增益的度量，其公式如下：
        \[ PE(s, x) = - \log p(s|x) = \sum_{i=1}^{N} - \log p(z_i|s<i, x) \]
        其中，\( s \) 是黄金响应，\( p(z_i|s<i, x) \) 是在黄金响应的第 \( i \) 个标记的概率。
        因此，ME和PE都用于评估模型预测的不确定性，但ME独立于真实标签，而PE考虑了真实标签的影响。
     
   - **动态不确定性测量(Dynamic Uncertainty Measurement)：**
     ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/152198b7-f797-41fd-ae2f-fc77c9654fc4)
     简单来说 样本不确定性的测量的指标由基础模型跟当前微调模型加权得到，随着迭代次数提高，基础模型权重更低，而当前微调模型权重更高

   - **混合正则化方法进行模型校准(Calibration with Hybrid Regularization):**
     对零初始化B矩阵，使用L2范数权重衰减
     ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/08beaba8-081c-4e1e-be28-c0cf30a41254)

     对高斯随机初始化的A矩阵，使用蒙特卡洛dropout机制
     ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/48d8756d-1000-476a-ac84-52a2a573abf6)


      
 

       
- 





### L2D(Learning_to_defer)

  



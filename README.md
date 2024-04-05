

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
     
        ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/90d762cf-ad0a-45fa-a6ea-b41ef2e1408c)
        ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/cf61142d-7378-48eb-98be-237f4639e055)


       
     
   - **动态不确定性测量(Dynamic Uncertainty Measurement)：**
     ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/152198b7-f797-41fd-ae2f-fc77c9654fc4)
     简单来说 样本不确定性的测量的指标由基础模型跟当前微调模型加权得到，随着迭代次数提高，基础模型权重更低，而当前微调模型权重更高

   - **混合正则化方法进行模型校准(Calibration with Hybrid Regularization):**
     对零初始化B矩阵，使用L2范数权重衰减
     ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/08beaba8-081c-4e1e-be28-c0cf30a41254)

     对高斯随机初始化的A矩阵，使用蒙特卡洛dropout机制
     ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/48d8756d-1000-476a-ac84-52a2a573abf6)

 - `NAACL 24` **From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning** [[paper](https://arxiv.org/pdf/2308.12032.pdf)][[code](https://github.com/tianyi-lab/Cherry_LLM)][Ming Li1, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, Jing Xiao] **这篇论文不是严格意义上的主动学习样本选择的论文(你可以理解为这篇论文的active learning迭代次数是2），但是其中的内核跟主动学习基于不确定性的方法大同小异，主要思想是提出一个指令随难度指标(Instruction-Following Difficulty，IFD),通过该指标来筛选具有增强LLM指令调优潜力的数据样例（樱桃数据，cherry data）**


   
   
   **总体方法:**
       ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/31dee965-aee1-42e8-b541-3599781f2fb7)
   
   - Learning from Brief Experience：利用少量进行进行模型初学：

        这个阶段的目标是通过强制模型首先体验目标数据集的一个子集来为初始模型提供基本的指令遵循能力。具体来说，对于初始的完整目标数据集 $D_0$ 包含 $n$ 个三元组 $x = (\text{Instruction}, [\text{Input}], \text{Answer})$，我们定义字符串 $Question =             \text{map(Instruction, [Input])}$ 作为完整指令。map 函数与原始目标数据集对齐。然后，对于每个样本 $x_j$，指令嵌入通过以下方式获得：
        
        $$[h^Q_{j,1}, \ldots, h^Q_{j,m}] = {LLM}_{\theta_{0}}(w^Q_{j,1}, \ldots, w^Q_{j,m})$$
   
        $$h^Q_{j} = \frac{1}{m} \sum_{i=1}^{m} h_{Qj,i}$$


        其中 $w^Q_{j,i}$ 表示样本 $j$ 的指令字符串的第 $i$ 个单词，$h^Q_{j,i}$ 表示其对应的最后隐藏状态。为了确保初始模型暴露给多样化的指令，利用基本的聚类技术 $KMeans$ 对这些指令嵌入进行聚类。受LIMA发现的启发，希望通过在每个聚类中仅采样少量实 
        例使这个体验过程尽可能简短。具体在指令嵌入上生成100个聚类，并在每个聚类中采样10个实例。然后，初始模型仅使用这些样本进行1个epoch的训练，以获得简要的预体验模型。
   

   
       - Evaluating Based on Experience：利用初学模型计算原始数据中所有IFD指标:
    
         以下是提取出的4个公式最终可以计算得出IFD值：
         
         ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/3f783cd9-53af-4a6a-b369-01499b8d3a61)
         ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/d1cc192a-39b7-413b-8164-367b1e15ba8c)
         ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/7ca9546c-3d66-490f-8996-54d29a331deb)
         ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/c9632a01-de8e-4db6-a8f7-be34817d8a2f)
         
         通过排序得到IFD最大的比例的数值对应的样本放入模型进行训练（IFD越高表示该prompt的难道越高，即模型越难完成该任务）

   **更多信息可以参考:https://zhuanlan.zhihu.com/p/664562587**

- `arXiv 23` **MoDS: Model-oriented Data Selection for Instruction Tuning** [[paper](https://arxiv.org/pdf/2311.15653.pdf)][[code](https://github.com/CASIA-LM/MoDS)[Qianlong Du, Chengqing Zong and Jiajun Zhang] **MoDS方法主要通过质量、覆盖范围、必要性三个指标来进行数据的筛选，其中数据质量是为了保证所选的指令数据的问题和答案都足够好；数据覆盖范围是为了让所选择的数据中指令足够多样、涉及知识范围更广；数据必要性是选择对于大模型较复杂、较难或不擅长的数据以填补大模型能力的空白。**
      ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/7ced16b0-1128-4b09-9289-373e6bcfbd22)

          - 质量筛选：对于数据进行质量过滤时，采用OpenAssistant的reward-model-debertav3-large-v2模型（一个基于DeBERTa架构设计的奖励模型）对数据进行质量打分。 讲原始数据的Instruction、Input、Output的三个部分进行拼接，送入到奖励模型中，得到一个评分，当评分超过α时，则认为数据质量达标，构建一份高质量数据集-Data1。
      
          - 多样性筛选：为了避免所选质量数据高度相似，通过K-Center-Greedy算法进行数据筛选，在最大化多样性的情况下，使指令数据集最小。获取种子指令数据集（Seed Instruction Data）-SID。
      
          - 必要性筛选：不同的大型语言模型在预训练过程中所学到的知识和具有的能力不同，因此在对不同的大型语言模型进行指令微调时，所需的指令数据也需要不同。
                -   对于一条指令，如果给定的大型语言模型本身能够生成较好的回答，则说明给定的大型语言模型具有处理该指令或者这类指令的能力，反之亦然，并且哪些不能处理的指令对于模型微调来说更为重要。
                -  使用SID数据集对模型进行一个初始训练
                -  用训练好的初始模型对整个高质数据集-Data1中的指令进行结果预测
                -  利用奖励模型对结果进行评分，当分值小于β时，说明初始模型不能对这些指令生成优质的回复，不具有处理这些类型指令的能力，获取必要性数据集-Data2
                -  对Data2进行多样性筛选，获取增强指令数据集（Augmented Instruction Data）-AID

          - 最终利用种子指令数据集和增强指令数据集一起对模型进行指令微调，获得最终模型。

- `ICLR 2024` **WHAT MAKES GOOD DATA FOR ALIGNMENT?A COMPREHENSIVE STUDY OF AUTOMATIC DATA SELECTION IN INSTRUCTION TUNING** [[paper](https://arxiv.org/pdf/2311.15653.pdf)][[code](https://github.com/hkust-nlp/deit)][Wei Liu, Weihao Zeng  Keqing He Yong Jiang Junxian He] **DEITA方法使用数据进行复杂性和质量评分，再通过多样性进行数据筛选。**

     - **总体架构:**

       ![image](https://github.com/yhr-code/ACTIVE_LEARNING_PAPER/assets/84458746/5ad1171c-9353-45bb-aa07-42ca652deb28)

       见:https://mp.weixin.qq.com/s/IqwP6cfsmPNduq_5Il7pow

- `EACL 23` **Investigating Multi-source Active Learning for Natural Language Infer**




### L2D(Learning_to_defer)

  



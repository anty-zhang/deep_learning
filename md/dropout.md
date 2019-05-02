
[TOC]

# dropout理解

## 组合派

1. 可以看成训练多个模型组合、效果平均的方式，即ensemble。dropout相当于从原始网络中国呢找到一个更瘦的网络。对于一个有n个节点的隐藏单元，有了dropout后可以堪称是$2^n$ 个模型的组合了，但此时训练的参数数目却是不变的。对于随机梯度下降来说，由于是随机失活隐藏单元，因此每个mini-batch都在训练不同的神经网络。
2. dropout使神经元的训练不依赖于另外一个神经元，特征之间的协同作用被减弱，这样可以避免过拟合
3. 从进化论的角度来解释。在自然界中，在中大型动物中，一般是有性繁殖，有性繁殖是指后代的基因从父母两方各继承一半。但是从直观上看，似乎无性繁殖更加合理，因为无性繁殖可以保留大段大段的优秀基因。而有性繁殖则将基因随机拆了又拆，破坏了大段基因的联合适应性。

但是自然选择中毕竟没有选择无性繁殖，而选择了有性繁殖，须知物竞天择，适者生存。我们先做一个假设，那就是基因的力量在于混合的能力而非单个基因的能力。不管是有性繁殖还是无性繁殖都得遵循这个假设。为了证明有性繁殖的强大，我们先看一个概率学小知识。

比如要搞一次恐怖袭击，两种方式： 

- 集中50人，让这50个人密切精准分工，搞一次大爆破。
 
- 将50人分成10组，每组5人，分头行事，去随便什么地方搞点动作，成功一次就算。

哪一个成功的概率比较大？ 显然是后者。因为将一个大团队作战变成了游击战。
<font color='red'>
那么，类比过来，有性繁殖的方式不仅仅可以将优秀的基因传下来，还可以降低基因之间的联合适应性，使得复杂的大段大段基因联合适应性变成比较小的一个一个小段基因的联合适应性。

dropout也能达到同样的效果，它强迫一个神经单元，和随机挑选出来的其他神经单元共同工作，达到好的效果。消除减弱了神经元节点间的联合适应性，增强了泛化能力。</font>

个人补充一点：那就是植物和微生物大多采用无性繁殖，因为他们的生存环境的变化很小，因而不需要太强的适应新环境的能力，所以保留大段大段优秀的基因适应当前环境就足够了。而高等动物却不一样，要准备随时适应新的环境，因而将基因之间的联合适应性变成一个一个小的，更能提高生存的概率。


## 噪声派

### 数据增强

对于每一个dropout后的网络，进行训练时，相当于做了Data Augmentation。因为，总是可以找到一个样本，使得在原始的网络上也能达到dropout单元后的结果。

比如对于每一层，dropout一些单元之后，形成了(1.5, 0, 2.5, 0, 1, 2, 0)，其中0是被dropout的单元，那么总能找到一个样本，使得结果也是如此。这样，每一次dropout其实都是相当于增加了样本。

### 稀疏性

- 在线性空间中，学习一个整个空间的特征集合是足够的，但是当数据分布在非线性不连续的空间中，则学习局部空间的特征组合会比较好。

> When the data points belonging to a particular class are distributed along a linear manifold, or sub-space, of the input space, it is enough to learn a single set of features which can span the entire manifold. But when the data is distributed along a highly non-linear and discontinuous manifold, the best way to represent such a distribution is to learn features which can explicitly represent small local regions of the input space, effectively “tiling” the space to define non-linear decision boundaries.


- dropout后，相当于得到了更多的局部簇，在同等数据下，簇变多了，因而可是使区分性变大，就使得稀疏性变大。

假设有一堆数据，这些数据由M个不同的非连续性簇表示，给定K个数据。那么一个有效的特征表示是将输入的每个簇映射为特征以后，簇之间的重叠度最低。使用A来表示每个簇的特征表示中激活的维度集合。重叠度是指两个不同的簇的Ai和Aj之间的Jaccard相似度最小，那么：

当K足够大时，即便A也很大，也可以学习到最小的重叠度
当K小M大时，学习到最小的重叠度的方法就是减小A的大小，也就是稀疏性。
上述的解释可能是有点太专业化，比较拗口。主旨意思是这样，我们要把不同的类别区分出来，就要是学习到的特征区分度比较大，在数据量足够的情况下不会发生过拟合的行为，不用担心。但当数据量小的时候，可以通过稀疏性，来增加特征的区分度。


# reference
https://blog.csdn.net/stdcoutzyx/article/details/49022443

[1]. Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: A simple way to prevent neural networks from overfitting[J]. The Journal of Machine Learning Research, 2014, 15(1): 1929-1958.

[2]. Dropout as data augmentation. http://arxiv.org/abs/1506.08700

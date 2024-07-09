
# Why Bert

Bert模型可以认为是最近两年NLP领域的集大成者。

- 从理论/模型创新角度看不太大，但效果好，基本刷新了NLP领域任务的最好性能

- 具备广泛的通用性。通过两个阶段方式去提升效果，绝大部分NLP任务都可以采用类似的两阶段模式直接提升效果。

## NLP 四类任务

- 序列标注: 分词/NER/语义标注

- 分类任务: 文本分类/情感分析

- 句子关系判断: Entailment/QA/自然语言推理

- 生成式任务: 机器翻译/文本摘要



## 预训练过程

### 图像领域的预处理 -- 类似于迁移学习

- 在A训练集任务上训练神经网络基础参数

- 在C 任务上通过两种方式来训练网络: （1）Frozen 基础参数不变（2）Fine-Tuning 基础参数进行微调

- 原因: 为什么这种预训练的方式可行？ （1）基础的网络学习到的是边角弧线等基础特征，和具体任务无关，具有通用性 （2）高层网络和具体的任务关联较大，比如轮廓等，可以通过Fine- tuning用新数据集清理掉高层无关的特征抽取器。

### Word Embedding历史

- NNLM（神经网络语言模型）。通过上文预测后面接的单词是什么; 同时能得到Word Embedding 副产品

- Word2Vec & Glove

两种训练方法：CBOW + Skip-Gram。

是单纯的训练Word Embedding。

- 缺点: 多义词的问题无法解决

### ELMO(Embedding from Language Models)

- Word Embedding本质上是个静态的方式，所谓静态指的是训练好之后每个单词的表达就固定住了

- ELMO的本质思想是：我事先用语言模型学好一个单词的Word Embedding，根据上下文单词的语义去调整单词的Word Embedding表示，这样经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。

- 缺点：（1）LSTM提取器能力弱于Transformer  （2）拼接方式双向融合特征能力偏弱

### GPT

- 其实和ELMO是类似的，主要不同在于两点：首先，特征抽取器不是用的RNN，而是用的Transformer，上面提到过它的特征抽取能力要强于RNN，这个选择很明显是很明智的；其次，GPT的预训练虽然仍然是以语言模型作为目标任务，但是采用的是单向的语言模型，所谓“单向”的含义是指：语言模型训练的任务目标是根据 W 单词的上下文去正确预测单词 W ，  之前的单词序列Context-before称为上文，之后的单词序列Context-after称为下文。

### Bert

- 和GPT的最主要不同在于在预训练阶段采用了类似ELMO的双向语言模型

- Bert最关键两点，一点是特征抽取器采用Transformer；第二点是预训练的时候采用双向语言模型。

[从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)

[放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)


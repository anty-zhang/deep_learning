


# 理论基础

## 前景理论
该理论解释了为什么人类在不确定的事件中做出的期望，不能使期望最大化，形式化了人类对损失更加敏感。PPO、DPO隐式地对这种偏差进行建模，被称为human aware loss function。

## 论文结论
1. 基于前景理论，KTO不需要一个数据对，只需要对生成的结果进行good/bad的二元标注即可。
2. 在1-30B的模型上，KTO的效果大于等于DPO。
3. KTO可以处理极端的数据不平衡问题，可以同时使用90%的好样本。
4. 当预训练模型足够好的时候，可以不对模型进行SFT，直接进行KTO。而DPO一定是需要SFT步骤的。


## 优势
1. 简单的反馈需求。data=(x, y, if_can_accept)，数据获取比较容易。
2. 在实践中对模型进行快速对齐。



# reference

[KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/pdf/2402.01306)

[KTO code](https://github.com/ContextualAI/HALOs)

[大模型对齐方法笔记一：DPO及其变种IPO、KTO、CPO](https://blog.csdn.net/beingstrong/article/details/138973997)



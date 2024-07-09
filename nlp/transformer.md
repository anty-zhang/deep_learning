
# Transformer 优势

- 多头 自注意力 机制

- 训练并行，提升训练效率

- 神经网络的深度可以足够深，充分发掘DNN的特性


# RNN 为何能成为NLP问题的主流特征抽取器

- NLP的输入往往是不定长的线性序列句子，RNN本身结构可以支持不定长输入 且 从前向后进行线性传到的网络结构

- LSTM/GRU 对于捕获长距离特征也是很有效

# CNN为何能成为NLP问题的特征抽取器

- k- gram片段是CNN捕获到的特征，K的大小决定了能捕获多远距离的特征

# 问题

- 为什么使用多头注意力机制

Transformer使用三角函数，使得计算远距离的位置关系之间的依赖关系的复杂度成为一个常数。但是，这种方式是通过平均注意力加权位置实现的会导致有效分辨率降低，所以可以使用多头注意力的机制来抵消这种影响。




[全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)

[Transformer读这一篇就够了](https://zhuanlan.zhihu.com/p/54356280)

[深度学习对话系统理论篇--seq2seq+Attention机制模型详解](https://zhuanlan.zhihu.com/p/32092871)

[Transformer 论文](https://arxiv.org/pdf/1706.03762.pdf)

[Transformer论文翻译](https://blog.csdn.net/qq_29695701/article/details/88096455)

[Transformer详解（三）：Transformer 结构](https://www.jianshu.com/p/0c196df57323)

[Attention机制详解（二）——Self-Attention与Transformer](https://zhuanlan.zhihu.com/p/47282410)

[图解Bert系列之Transformer实战 (附代码)](http://www.uml.org.cn/ai/2019101114.asp)

[Transformer原理以及文本分类实战](https://blog.csdn.net/qq_36618444/article/details/106472126)

[NLP预训练模型-Transformer：从原理到实战](https://blog.csdn.net/linxid/article/details/84321617)

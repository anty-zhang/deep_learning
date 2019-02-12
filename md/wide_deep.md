# reference
====

https://arxiv.org/pdf/1606.07792.pdf  (原始论文)
https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html
http://www.360doc.com/content/17/0204/14/15077656_626440878.shtml (谷歌新开源「宽度&深度学习」框架：实现更优推荐)
http://www.vccoo.com/v/k3nho5
http://geek.csdn.net/news/detail/235465 (TensorFlow Wide And Deep 模型详解与应用)

## 应用场景

> 带有稀疏输入的一般大规模回归和分类问题，推荐系统、搜索、排名问题

## wide and deep核心

> 核心思想是结合线性模型的记忆能力（memorization）和 DNN 模型的泛化能力（generalization），在训练过程中同时优化 2 个模型的参数，
从而达到整体模型的预测能力最优。

> 记忆用户来提供与主题相关的推荐

> 归纳用来探索从未或者很少过去出现的新的特征组合

## 人类记忆和归纳的学习方式

> 通过记忆日常中的种种事件(比如云雀会飞|企鹅会飞)形成规则， 并归纳这些学习应用到从未出现的事物上(比如有翅膀的动物会飞)

> 更强大的是记忆可以使我们使用例外(比如企鹅不会飞), 进一步提炼我们归纳的规则

## wide模型

> 可以记住哪些项目与每次查询匹配的效果最好

> 比如说，学习了特征 AND(query="fried chicken", item="chicken fried rice") （查询「炸鸡」，得到项目「炸鸡和华夫饼」）
的模型效果非常好；尽管  AND(query="fried chicken", item="chicken fried rice")（查询「炸鸡」，得到「炸鸡饭」）的匹配度
也很好但消费率不高。也就是说，FoodIO 2.0 可以很好地记忆用户的喜好，它开始对用户越来越有吸引力。

## deep 模型

> 深度模型用来解决： 当用户已经对推荐中的物品感觉到厌烦了，但渴望出现类似但又不同的眼前一亮的物品。

> 深度模型学习了每个查询和项目的低纬密集特征(embedding vector), 通过在embedding空间中匹配彼此比较近的项目或查询进行归纳

## wide & deep

> 通过deep可以解决用户已经厌烦的推荐物品， 又可以通过wide解决归纳出太多的和推荐不相关的物品

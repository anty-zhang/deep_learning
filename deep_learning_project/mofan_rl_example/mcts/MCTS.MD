
# 蒙特卡洛模拟

## 实例

### 模拟PI值

> 单位圆的面积是PI，单位圆的外切正方形面积是4. 所以，1/4个圆的面积是PI/4，外切正方形的面积是1. 而1/4圆的面积/单位正方形的面积就是PI/4

![模拟PI值](./img/pi_mcts.jpg)


# MCTS 蒙特卡洛树搜索



# 最有潜力的下一步行动 --> minimax策略和alpha-beta剪枝算法

## 极小极大算法弱点

> 需要展开整个博弈树，对于有高分支因子的博弈将会产生巨大的博弈树，使计算无法进行

## 解决办法

### 在确定的阀值深度d内展开博弈树

> 但是我们无法保证在阈值深度 d 处的任何节点是否端节点。因此我们一个函数来评估非终端博弈状态。这对于人类来说很自然：即使博弈仍在进行，你也可能通过
观察围棋或国际象棋的棋盘预测胜者

### alpha-beta 剪枝

> 通过alpha-beta剪枝算法来修剪博弈树。alpha-beta 剪枝是提升版的极小极大算法，它以极小极大算法的形式遍历博弈树，并避免某些树分支的展开，其得到
的结果在最好的情况下等于极小极大算法的结果。alpha-beta 剪枝通过压缩搜索空间提高搜索效率。



https://blog.csdn.net/ljyt2/article/details/78332802?locationNum=9&fps=1 （蒙特卡洛树搜索（MCTS）算法）

https://cloud.tencent.com/developer/article/1150410 （mcts最新成果）

https://www.jiqizhixin.com/articles/monte-carlo-tree-search-beginners-guide （）









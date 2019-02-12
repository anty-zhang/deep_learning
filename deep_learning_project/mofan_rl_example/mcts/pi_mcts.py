# -*- coding:utf-8 -*-
"""
使用蒙特卡罗方法计算PI值
"""

import random
import math
import time

# 调用一次perf_counter()，从计算机系统里随机选一个时间点A，计算其距离当前时间点B1有多少秒。当第二次调用该函数时，
# 默认从第一次调用的时间点A算起，距离当前时间点B2有多少秒。两个函数取差，即实现从时间点B1到B2的计时功能。
start_time = time.perf_counter()

total = 10000 * 10000
hits = 0

for c in range(total):
    x = random.random()
    y = random.random()

    if math.sqrt(x**2 + y**2) <= 1:
        hits += 1

PI = 4 * (hits/total)

print("PI=", PI)

end_time = time.perf_counter()

print("COST {:.2f}S".format(end_time - start_time))


"""
result:
PI= 3.14158952
COST 52.55S
"""
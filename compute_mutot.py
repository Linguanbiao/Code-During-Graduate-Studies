from scipy.stats import norm
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import math


def MU_TOT(x):
    y = norm.cdf((-(4.5) / x) + (x / 2)) - (math.exp(4.5) * norm.cdf((-4.5 / x) - (x / 2))) - 1e-5
    return y


def MU_0(x, mu_tot):
    sum_term = 0
    for t in range(1, 2344):
        a = t / 2344
        sum_term += (math.exp(((2) ** a * x) ** 2) - 1)
    y = (1024 / 60000) ** 2 * sum_term - mu_tot ** 2
    return y


left_x_tot = 0.01
right_x_tot = 10
target_x_tot = 0

while True:
    y1 = MU_TOT(left_x_tot)
    y2 = MU_TOT(right_x_tot)
    if abs(y1) < 1e-6:
        target_x_tot = left_x_tot
        print('target_x_tot', target_x_tot)
        break
    elif abs(y2) < 1e-6:
        target_x_tot = right_x_tot
        print('target_x_tot', target_x_tot)
        break
    else:
        target_x_tot = (left_x_tot + right_x_tot) / 2
        if MU_TOT(target_x_tot) > 0:
            right_x_tot = target_x_tot
        else:
            left_x_tot = target_x_tot

left_x_0 = 0.01
right_x_0 = 5
target_x_0 = 0


while True:
    y1 = MU_0(left_x_0, target_x_tot)
    y2 = MU_0(right_x_0, target_x_tot)
    if abs(y1) < 1e-6:
        target_x_0 = left_x_0
        print('target_x_0', target_x_0)
        break
    elif abs(y2) < 1e-6:
        target_x_0 = right_x_0
        print('target_x_0', target_x_0)
        break
    else:
        target_x_0 = (left_x_0 + right_x_0) / 2
        if MU_0(target_x_0, target_x_tot) > 0:
            right_x_0 = target_x_0
        else:
            left_x_0 = target_x_0

# z = norm.cdf((-(130.5) / 12.482148437499998) + (12.482148437499998 / 2)) - (math.exp(130.5) * norm.cdf((-130.5 / 12.482148437499998) - (12.482148437499998 / 2)))
# print(z)


# while True:
#     y1 = MU_0(left_x)
#     y2 = MU_0(right_x)
#     if abs(y1) < 1e-6:
#         target_x = left_x
#         print('target_x', target_x)
#         break
#     elif abs(y2) < 1e-6:
#         target_x = right_x
#         print('target_x', target_x)
#         break
#     else:
#         target_x = (left_x + right_x) / 2
#         if MU_0(target_x) > 0:
#             right_x = target_x
#         else:
#             left_x = target_x

# x = np.array([0.2458984375])
# y = norm.cdf((-(0.91) / x) + (x / 2)) - (math.exp(0.91) * norm.cdf((-0.91 / x) - (x / 2)))
# print(y)

# y = pow(x, 1.0/2)

# plt.figure() # 定义一个图像窗口
# plt.plot(x, y) # 绘制曲线 y
# plt.xlabel('Epochs')
# plt.ylabel('Clip_values')

# plt.legend()
# plt.legend()
# plt.savefig('pictures_3.png')

# plt.show()

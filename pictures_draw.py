import matplotlib.pyplot as plt
import numpy as np
import math 

x = np.linspace(1, 40, 40) 
y = 2.0 / min(2, 1 + x / 40)
# y = pow(x, 1.0/2)

# plt.figure() # 定义一个图像窗口
plt.plot(x, y) # 绘制曲线 y
plt.xlabel('Epochs')
plt.ylabel('Clip_values')

# plt.legend()
# plt.legend()
plt.savefig('./picture/pictures3.png')

# plt.show()



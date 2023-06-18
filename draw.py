import numpy as np
import matplotlib.pyplot as plt
import math

# if __name__ == '__main__':
#     load_dict_gn = np.load('./cifar10_result/sigma_0.91_epoch_40_dec_2.5_3.npy', allow_pickle=True).item()
#     load_dict = np.load('./cifar10_result/sigma_0.91_epoch_40_flat_0.6_1.npy', allow_pickle=True).item()
#     load_dict_dp_gn = np.load('./cifar10_result/sigma_0.91_epoch_40_flat_1.2_1.npy', allow_pickle=True).item()
#     # load_dict_dp = np.load('./cifar10_result/sigma_0.90_epoch_40_flat_1.5.npy', allow_pickle=True).item()
#     # load_dict_dp_all = np.load('./MNIST/mnist_per_vision_1.0_change_dec.npy', allow_pickle=True).item()
#     # loda_dict_dp_w = np.load('./model_100/mnist_Flat_vision_adam.npy',allow_pickle=True).item()
#     #
#     x1 = list(load_dict_gn.keys())
#     y1 = list(load_dict_gn.values())
#     #
#     x2 = list(load_dict.keys())
#     y2 = list(load_dict.values())

#     x3 = list(load_dict_dp_gn.keys())
#     y3 = list(load_dict_dp_gn.values())
#     #
#     # x4 = list(load_dict_dp.keys())
#     # y4 = list(load_dict_dp.values())
#     #
#     # x5 = list(load_dict_dp_all.keys())
#     # y5 = list(load_dict_dp_all.values())
#     #
#     # x6 = list(loda_dict_dp_w.keys())
#     # y6 = list(loda_dict_dp_w.values())

#     plt.plot(x1, y1, label='dec', color='r', linewidth=1)
#     plt.plot(x2, y2, label='flat_0.6', color='b', linewidth=1)
#     plt.plot(x3, y3, label='flat_1.2', color='c', linewidth=1)
#     # plt.plot(x4, y4, label='flat_1.5', color='y', linewidth=1)
#     # plt.plot(x5, y5, label='per_change_bias=0.2', color='r', linewidth=3)
#     # plt.plot(x6, y6, label='Flat_adam',color='c',linewidth=3)

#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')

#     # plt.yticks(0.4,1,0.05)
#     # plt.ylim(0, 1)
#     # y = np.linspace(0.4, 1.0, 12)
#     # plt.yticks(y, fontsize=9)
#     plt.legend()
#     plt.savefig('./picture/cifar10/sigma_0.91.png')
#     # plt.show()

print(math.pow(0.9 , 3))
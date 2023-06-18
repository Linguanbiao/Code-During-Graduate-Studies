import numpy as np
import json
from torchvision import datasets
import matplotlib.pyplot as plt

with open('fashionmnist_lr1_batchsize1024_sigma1.47.json', 'r', encoding='utf8') as fp:
    json_data_1 = json.load(fp)

x = [i for i in range(0, 40)]
y = []

for i in json_data_1:
    y.append(json_data_1[i][0])

x = np.array(x)
y = np.array(y)
#
parameter = np.polyfit(x, y, 3)
y2 = parameter[0] * x ** 3 + parameter[1] * x ** 2 + parameter[2] * x ** 1 + parameter[3]
print(parameter)
#
plt.plot(x, y,  'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='ADA')
plt.plot(x, y2, color='g')
plt.savefig('fashionmnist_function_1.47.png')

# data_path = 'dataset'
# train = datasets.FashionMNIST(data_path, train=True, download=True)
# test = datasets.FashionMNIST(data_path, train=False, download=False)
# x, y = train.data.numpy(), train.targets.numpy()
# xTest, yTest = test.data.numpy(), test.targets.numpy()
# x = x[:, np.newaxis, :, :]
# xTest = xTest[:, np.newaxis, :, :]
# print(x.shape, xTest.shape)


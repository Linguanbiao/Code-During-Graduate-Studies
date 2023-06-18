# import torch
# from torch import nn
# # from torchsummary import summary


# class cifar10_model(nn.Module):
#     def __init__(self):
#         super(cifar10_model, self).__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(start_dim=1, end_dim=-1),
#             nn.Linear(128, 10, bias=True),
#         )

#     def forward(self, x):
#         out = self.network(x)
#         return out


# if __name__ == '__main__':
#     net = cifar10_model().cuda()
#     summary(net, (3, 32, 32))


import torch
from torch import nn
# from torchsummary import summary


class cifar10_model(nn.Module):
    def __init__(self):
        super(cifar10_model, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.network(x)
        return out


if __name__ == '__main__':
    net = cifar10_model().cuda()
    summary(net, (3, 32, 32))

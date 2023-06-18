import torch
from torch import nn
from torchsummary import summary

# class cnn2(nn.Module):
#     def __init__(self):
#         super(cnn2, self).__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(1, 16, (8, 8), padding=(17, 17), stride=(2, 2)),
#             nn.Tanh(),
#             nn.MaxPool2d((2, 2), (1, 1)),
#             nn.Conv2d(16, 32, (4, 4), stride=(2, 2)),
#             nn.Tanh(),
#             nn.MaxPool2d((2, 2), (1, 1)),
#             nn.Conv2d(32, 32, (4, 4), stride=(2, 2)),
#             nn.Tanh(),
#             nn.MaxPool2d((2, 2), (1, 1)),
#             nn.Flatten(),
#             nn.Linear(288, 2),
#             nn.Tanh(),
#             nn.Linear(2, 10)
#         )

#     def forward(self, x):
#         out = self.network(x)
#         return out


# if __name__ == '__main__':
#     net = cnn2().cuda()
#     summary(net, (1, 28, 28))


from torch import nn
from torchsummary import summary


class BadNet(nn.Module):

    def __init__(self):
        super(BadNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, 5)
        # self.conv2 = nn.Conv2d(16, 32, 5)
        # self.pool = nn.AvgPool2d(2)
        # self.fc1 = nn.Linear(512, 512)
        # self.fc2 = nn.Linear(512, 10)
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            # nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            # nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.network(x)
        return out
        # x = self.conv1(x)
        # # x = F.relu(x)
        # x = F.tanh(x)
        # x = self.pool(x)
        # x = self.conv2(x)
        # # x = F.relu(x)
        # x = F.tanh(X)
        # x = self.pool(x)
        # x = x.view(-1, self.num_f(x))
        # x = self.fc1(x)
        # # x = F.relu(x)
        # x = F.tanh(X)
        # x = self.fc2(x)
        # x = F.softmax(x,dim = 1)

    # def num_f(self, x):
    #     size = x.size()[1:]
    #     ret = 1
    #     for i in size:
    #         ret *= i
    #     return ret


if __name__ == '__main__':
    net = BadNet().cuda()
    summary(net, (1, 28, 28))

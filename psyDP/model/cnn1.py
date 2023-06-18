import torch
from torch import nn
from torchsummary import summary


class cnn1(nn.Module):
    def __init__(self):
        super(cnn1, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, (8, 8), padding=(17, 17), stride=(2, 2)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2), (1, 1)),
            nn.Conv2d(16, 32, (4, 4), stride=(2, 2)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2), (1, 1)),
            nn.Flatten(),
            nn.Linear(3872, 32),
            nn.Tanh(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        out = self.network(x)
        return out


if __name__ == '__main__':
    net = cnn().cuda()
    summary(net, (1, 28, 28))

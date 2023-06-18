from torch import nn
from torchsummary import summary


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.Tanh(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.Tanh(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Tanh(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Tanh(),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = AlexNet().cuda()
    summary(net, (3, 224, 224))

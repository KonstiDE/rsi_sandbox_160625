import torch
import torch.nn as nn

class RSIModel(nn.Module):
    def __init__(self, out_classes=4):
        super(RSIModel, self).__init__()

        self.initial = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1)

        self.blue = QuadConvDoubleRes(64, 64, 3, stride_first=False)
        self.green = QuadConvDoubleRes(64, 128, 3, stride_first=True)
        self.orange = QuadConvDoubleRes(128, 256, 3, stride_first=True)
        self.grey = QuadConvDoubleRes(256, 512, 3, stride_first=True)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_classes)

    def forward(self, x):
        x = self.initial(x)

        x = self.blue(x)
        x = self.green(x)
        x = self.orange(x)
        x = self.grey(x)

        x = self.avg(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class QuadConvDoubleRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_first):
        super(QuadConvDoubleRes, self).__init__()
        self.stride_first = stride_first

        stride = 2 if stride_first else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)

        self.relu = nn.ReLU(inplace=False)

        self.skip = nn.Identity()
        if stride_first or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = x + residual
        residual = x.clone()

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = x + residual

        return x


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    model = RSIModel(out_classes=4)

    a = torch.randn((1, 3, 256, 256))  # Include batch size = 1
    print(model(a).shape)  # Should output: torch.Size([1, 4])



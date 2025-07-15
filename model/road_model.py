import torch
import torch.nn as nn

from model.road_layers import (
    DoubleConvolution,
    UpConvolution
)

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.skips = []
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dc1 = DoubleConvolution(3, 64)
        self.dc2 = DoubleConvolution(64, 128)
        self.dc3 = DoubleConvolution(128, 256)
        self.dc4 = DoubleConvolution(256, 512)
        self.dc5 = DoubleConvolution(512, 1024)

        self.uc1 = UpConvolution(in_channels=1024, out_channels=512)
        self.dc1butup = DoubleConvolution(1024, 512)

        self.uc2 = UpConvolution(in_channels=512, out_channels=256)
        self.dc2butup = DoubleConvolution(512, 256)

        self.uc3 = UpConvolution(in_channels=256, out_channels=128)
        self.dc3butup = DoubleConvolution(256, 128)

        self.uc4 = UpConvolution(in_channels=128, out_channels=64)
        self.dc4butup = DoubleConvolution(128, 64)

        self.final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.dc1(x)
        self.skips.append(x)
        x = self.pool(x)

        x = self.dc2(x)
        self.skips.append(x)
        x = self.pool(x)

        x = self.dc3(x)
        self.skips.append(x)
        x = self.pool(x)

        x = self.dc4(x)
        self.skips.append(x)
        x = self.pool(x)

        x = self.dc5(x)


        #### Decoder begins here ####
        x = self.uc1(x)
        x = torch.cat((self.skips[-1], x), dim=1)
        x = self.dc1butup(x)

        x = self.uc2(x)
        x = torch.cat((self.skips[-2], x), dim=1)
        x = self.dc2butup(x)

        x = self.uc3(x)
        x = torch.cat((self.skips[-3], x), dim=1)
        x = self.dc3butup(x)

        x = self.uc4(x)
        x = torch.cat((self.skips[-4], x), dim=1)
        x = self.dc4butup(x)

        return self.final(x)



if __name__ == '__main__':
    a = torch.randn((8, 3, 256, 256))

    model = UNET()

    print(model(a).shape)



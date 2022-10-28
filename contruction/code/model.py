import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, residual=False, down=False):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.residual = residual
        self.down = down

        # self.conv = nn.Conv2d(in_channels, out_channels, (3,3), (1,1), 1, bias= False)
        # self.bn = nn.BatchNorm2d()

    def forward(self, x):
        if self.residual:
            return self.conv(x) + self.conv_skip(x)

        return self.conv(x)


class UNET(nn.Module):
    def __init__(self
                 , in_channels=3
                 , out_channels=1
                 , features=[16, 32, 64, 128, 256]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoding parts
        for idx, feature in enumerate(features):
            if idx in [3, 4]:
                drop_out = 0.5
            else:
                drop_out = 0
            self.downs.append(DoubleConv(in_channels, feature, down=True, dropout=drop_out, residual=True))
            in_channels = feature

        # decoding parts
        for idx, feature in enumerate(reversed(features)):
            if idx in [0, 1]:
                drop_out = 0.5
            else:
                drop_out = 0

            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=3, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature, dropout=drop_out, residual=True))

        # TODO: modify bottleneck to similar as paper
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=0.5)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):

            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            # double conv
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((1, 3, 512, 512))
    # torch.reshape(x, [3, 512, 512])
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    # assert preds.shape == x.shape


if __name__ == "__main__":
    # model = UNET(in_channels=1, out_channels=1)
    # print(model)
    test()

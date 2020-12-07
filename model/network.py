import torch


class ConvLayer(torch.nn.Module):
    def __init__(self, in_planes, out_planes, down_sample=True):
        super(ConvLayer, self).__init__()
        if down_sample:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_planes),
                torch.nn.ReLU(inplace=True)
            )

        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_planes),
                torch.nn.ReLU(inplace=True)
            )

    def forward(self, x):
        out = self.conv(x)
        return out


class DeconvLayer(torch.nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DeconvLayer, self).__init__()
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=2,
                                     stride=2, padding=0, output_padding=0, bias=False),
            torch.nn.BatchNorm2d(out_planes),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.deconv(x)
        return out


class ResidualLayer(torch.nn.Module):
    def __init__(self, in_planes):
        super(ResidualLayer, self).__init__()
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1),
            torch.nn.BatchNorm2d(in_planes),
            torch.nn.ReLU()
        )

        self.shortcut = torch.nn.Sequential()

    def forward(self, x):
        out = self.bottleneck(x)
        out = out + self.shortcut(x)
        return out


class UNet(torch.nn.Module):
    def __init__(self, channels=3):
        super(UNet, self).__init__()
        self.down_layers = [
            ConvLayer(channels, 8),
            ConvLayer(8, 16),
            ConvLayer(16, 32),
            ConvLayer(32, 64),
            ConvLayer(64, 128)
        ]

        self.deconv_layers = [
            DeconvLayer(128, 64),
            DeconvLayer(64, 32),
            DeconvLayer(32, 16),
            DeconvLayer(16, 8),
            DeconvLayer(8, 1)
        ]

        self.conv_wo_downsamp = [
            ConvLayer(128, 64, down_sample=False),
            ConvLayer(64, 32, down_sample=False),
            ConvLayer(32, 16, down_sample=False),
            ConvLayer(16, 8, down_sample=False)
        ]

        self.residual_layers = [
            ResidualLayer(64),
            ResidualLayer(32),
            ResidualLayer(16),
            ResidualLayer(8)
        ]

        self.down_layers = torch.nn.ModuleList(self.down_layers)
        self.deconv_layers = torch.nn.ModuleList(self.deconv_layers)
        self.conv_wo_downsamp = torch.nn.ModuleList(self.conv_wo_downsamp)
        self.residual_layers = torch.nn.ModuleList(self.residual_layers)

    def forward(self, x):
        down_outputs = []
        for down_layer in self.down_layers:
            x = down_layer(x)
            down_outputs.insert(0, x)

        up_outputs = down_outputs[0]
        for i in range(len(down_outputs) - 1):
            deconv_cur_layer = self.deconv_layers[i](up_outputs)
            residual_cur_up_layer = self.residual_layers[i](down_outputs[i + 1])
            concate_cur_layers = torch.cat((deconv_cur_layer, residual_cur_up_layer), dim=1)
            conv_after_concat = self.conv_wo_downsamp[i](concate_cur_layers)
            up_outputs = conv_after_concat

        up_outputs = self.deconv_layers[-1](up_outputs)
        return up_outputs


if __name__ == '__main__':
    net = UNet()
    y = net(torch.randn(2, 3, 256, 256))
    print(y.size())

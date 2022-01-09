import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from module.Local_Aggregation import Local_Aggregation
from module.Global_Aggregation import Global_Aggregation


class stcsa(nn.Module):
    def __init__(self, feat_num):
        super(stcsa, self).__init__()
        self.attention = Global_Aggregation(in_channels=feat_num)

    def forward(self, x):  # torch.Size([1, 25, 96, 128, 128])
        x = self.attention(x)
        return x


class Net(nn.Module):
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        n_blocks, channel = 4, 36
        self.factor = factor
        self.angRes = angRes
        self.FeaExtract = FeaExtract(channel)
        self.ADAM_1 = Local_Aggregation(channel, angRes)
        self.ADAM_2 = Local_Aggregation(channel, angRes)
        self.trans_row = stcsa(channel)
        self.trans_col = stcsa(channel)

        self.Reconstruct = CascadedBlocks(n_blocks, 5 * channel)
        self.UpSample = Upsample(channel, factor)

    def forward(self, x):  # torch.Size([1, 1, 640, 640])
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bicubic',
                                  align_corners=False)  # torch.Size([1, 1, 2560, 2560])
        x = LFsplit(x, self.angRes)

        buffer_0 = self.FeaExtract(x)
        buffer_1 = self.ADAM_1(buffer_0)
        buffer_2 = self.ADAM_2(buffer_1)

        buffer_row = []
        for i in range(5):
            row = buffer_2[:, 5 * i:5 * (i + 1)]
            Tran_row = self.trans_row(row)
            buffer_row.append(Tran_row)
        buffer_row = torch.cat(buffer_row, dim=1)

        buffer_col = []
        for i in range(5):
            col = []
            for j in range(5):
                col.append(buffer_2[:, 5 * j + i].unsqueeze(1))
            col = torch.cat(col, dim=1)
            Tran_col = self.trans_col(col)
            buffer_col.append(Tran_col)
        buffer_col = torch.cat(buffer_col, dim=1)
        buffer_col = Col_T(buffer_col)

        buffer = torch.cat((buffer_0, buffer_1, buffer_2, buffer_row, buffer_col), dim=2)
        buffer = self.Reconstruct(buffer)  # torch.Size([1, 24, 128, 128, 128])
        out = self.UpSample(buffer)  # torch.Size([1, 24, 1, 512, 512])
        out = FormOutput(out) + x_upscale  # torch.Size([1, 1, 2560, 2560])

        return out


def Col_T(feature):  # B N C H W
    feature_T = []
    for i in range(5):
        col = []
        for j in range(5):
            col.append(feature[:, 5 * j + i].unsqueeze(1))
        col = torch.cat(col, dim=1)
        feature_T.append(col)
    feature_T = torch.cat(feature_T, dim=1)  # 2 25 96 128 128
    return feature_T


class Upsample(nn.Module):
    def __init__(self, channel, factor):
        super(Upsample, self).__init__()
        self.upsp = nn.Sequential(
            nn.Conv2d(5 * channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b * n, -1, h, w)
        out = self.upsp(x)
        _, _, H, W = out.shape
        out = out.contiguous().view(b, n, -1, H, W)
        return out


class FeaExtract(nn.Module):
    def __init__(self, channel):
        super(FeaExtract, self).__init__()
        self.FEconv = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.FERB_1 = ResASPP(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = ResASPP(channel)
        self.FERB_4 = RB(channel)

    def forward(self, x):
        b, n, h, w = x.shape
        x = x.contiguous().view(b * n, -1, h, w)
        buffer_x_0 = self.FEconv(x)
        buffer_x = self.FERB_1(buffer_x_0)
        buffer_x = self.FERB_2(buffer_x)
        buffer_x = self.FERB_3(buffer_x)
        buffer_x = self.FERB_4(buffer_x)
        _, c, h, w = buffer_x.shape
        buffer_x = buffer_x.unsqueeze(1).contiguous().view(b, -1, c, h, w)  # buffer_sv:  B, N, C, H, W

        return buffer_x


class CascadedBlocks(nn.Module):
    def __init__(self, n_blocks, channel):
        super(CascadedBlocks, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(IMDB(channel))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.body[i](x)
        return x


class RB(nn.Module):
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x


class IMDB(nn.Module):
    def __init__(self, channel):
        super(IMDB, self).__init__()
        self.conv_0 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_1 = nn.Conv2d(3 * channel // 4, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(3 * channel // 4, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(3 * channel // 4, channel // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv_t = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer = x.contiguous().view(b * n, -1, h, w)
        buffer = self.lrelu(self.conv_0(buffer))
        buffer_1, buffer = ChannelSplit(buffer)
        buffer = self.lrelu(self.conv_1(buffer))
        buffer_2, buffer = ChannelSplit(buffer)
        buffer = self.lrelu(self.conv_2(buffer))
        buffer_3, buffer = ChannelSplit(buffer)
        buffer_4 = self.lrelu(self.conv_3(buffer))
        buffer = torch.cat((buffer_1, buffer_2, buffer_3, buffer_4), dim=1)
        buffer = self.lrelu(self.conv_t(buffer))
        x_buffer = buffer.contiguous().view(b, n, -1, h, w)
        return x_buffer + x


def ChannelSplit(input):
    _, C, _, _ = input.shape
    c = C // 4
    output_1 = input[:, :c, :, :]
    output_2 = input[:, c:, :, :]
    return output_1, output_2


class ResASPP(nn.Module):
    def __init__(self, channel):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1,
                                              dilation=1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2,
                                              dilation=2, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=4,
                                              dilation=4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv2d(channel * 3, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_out = []
    for u in range(angRes):
        for v in range(angRes):
            data_out.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])

    data_out = torch.cat(data_out, dim=1)
    return data_out


def FormOutput(x_sv):
    b, n, c, h, w = x_sv.shape
    angRes = int(sqrt(n + 1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk + 1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


if __name__ == "__main__":
    net = Net(5, 4).cuda()
    from thop import profile

    input = torch.randn(1, 1, 160, 160).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))

#    Number of parameters: 3.78M
#    Number of FLOPs: 113.34G

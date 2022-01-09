import torch
import torch.nn as nn
from dcn.modules.deform_conv import DeformConv


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


class Local_Aggregation(nn.Module):
    def __init__(self, channel, angRes):
        super(Local_Aggregation, self).__init__()
        self.conv_1 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.ASPP = ResASPP(channel)
        self.conv_off = nn.Conv2d(channel, 2 * 9, kernel_size=1, stride=1, padding=0)
        self.conv_off.lr_mult = 0.1
        self.init_offset()

        self.conv_f3_1 = nn.Conv2d(4 * channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv_f5_1 = nn.Conv2d(6 * channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv_f8_1 = nn.Conv2d(9 * channel, channel, kernel_size=1, stride=1, padding=0)

        self.dcn = DeformConv(channel, channel, 3, 1, 1, deformable_groups=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def init_offset(self):
        self.conv_off.weight.data.zero_()
        self.conv_off.bias.data.zero_()

    def forward(self, feature):
        out = []
        for num in range(25):
            sv, cv = self.split_sv_cv(feature, num)
            b, n, c, h, w = sv.shape
            aligned_fea = []
            for i in range(n):
                current_sv = sv[:, i, :, :, :].contiguous()
                buffer = torch.cat((current_sv, cv), dim=1)  # B * 2C * H * W
                buffer = self.lrelu(self.conv_1(buffer))  # B * C * H * W
                buffer = self.ASPP(buffer)  # B * C * H * W
                offset = self.conv_off(buffer)  # B, 18, H, W
                current_aligned_fea = self.lrelu(self.dcn(current_sv, offset))  # B, C, H, W
                aligned_fea.append(current_aligned_fea)
            aligned_fea = torch.cat(aligned_fea, dim=1)  # B, N*C, H, W
            fea_collect = torch.cat((cv, aligned_fea), 1)  # B, (N+1)*C, H, W

            if (n == 3):
                fuse_fea = self.conv_f3_1(fea_collect)  # B, (N+1)*C, H, W
            if (n == 5):
                fuse_fea = self.conv_f5_1(fea_collect)  # B, (N+1)*C, H, W
            if (n == 8):
                fuse_fea = self.conv_f8_1(fea_collect)  # B, (N+1)*C, H, W

            out_cv = cv + fuse_fea
            out_cv = out_cv.unsqueeze(1)
            out.append(out_cv)
        out = torch.cat(out, dim=1)
        return out

    def split_sv_cv(self, feature, num):  # b, n, c, h, w
        if num in [0]:  # Top-left
            cv = feature[:, num]
            sv = torch.cat(
                (feature[:, num + 1].unsqueeze(1), feature[:, num + 5].unsqueeze(1), feature[:, num + 6].unsqueeze(1)),
                dim=1)
            return sv, cv

        if num in [1, 2, 3]:  # Top
            cv = feature[:, num]
            sv = torch.cat(
                (feature[:, num - 1].unsqueeze(1), feature[:, num + 1].unsqueeze(1), feature[:, num + 4].unsqueeze(1),
                 feature[:, num + 5].unsqueeze(1), feature[:, num + 6].unsqueeze(1)),
                dim=1)
            return sv, cv

        if num in [4]:  # Top-right
            cv = feature[:, num]
            sv = torch.cat(
                (feature[:, num - 1].unsqueeze(1), feature[:, num + 4].unsqueeze(1), feature[:, num + 5].unsqueeze(1)),
                dim=1)
            return sv, cv

        if num in [5, 10, 15]:  # Left
            cv = feature[:, num]
            sv = torch.cat(
                (feature[:, num - 5].unsqueeze(1), feature[:, num - 4].unsqueeze(1), feature[:, num + 1].unsqueeze(1),
                 feature[:, num + 5].unsqueeze(1), feature[:, num + 6].unsqueeze(1)),
                dim=1)
            return sv, cv

        if num in [6, 7, 8, 11, 12, 13, 16, 17, 18]:  # Middle
            cv = feature[:, num]
            sv = torch.cat(
                (feature[:, num - 6].unsqueeze(1), feature[:, num - 5].unsqueeze(1), feature[:, num - 4].unsqueeze(1),
                 feature[:, num - 1].unsqueeze(1), feature[:, num + 1].unsqueeze(1), feature[:, num + 4].unsqueeze(1),
                 feature[:, num + 5].unsqueeze(1), feature[:, num + 6].unsqueeze(1)),
                dim=1)
            return sv, cv

        if num in [9, 14, 19]:  # Right
            cv = feature[:, num]
            sv = torch.cat(
                (feature[:, num - 6].unsqueeze(1), feature[:, num - 5].unsqueeze(1), feature[:, num - 1].unsqueeze(1),
                 feature[:, num + 4].unsqueeze(1), feature[:, num + 5].unsqueeze(1)),
                dim=1)
            return sv, cv

        if num in [20]:  # Bottom-left
            cv = feature[:, num]
            sv = torch.cat(
                (feature[:, num - 5].unsqueeze(1), feature[:, num - 4].unsqueeze(1), feature[:, num + 1].unsqueeze(1)),
                dim=1)
            return sv, cv

        if num in [21, 22, 23]:  # Bottom
            cv = feature[:, num]
            sv = torch.cat(
                (feature[:, num - 6].unsqueeze(1), feature[:, num - 5].unsqueeze(1), feature[:, num - 4].unsqueeze(1),
                 feature[:, num - 1].unsqueeze(1), feature[:, num + 1].unsqueeze(1)),
                dim=1)
            return sv, cv

        if num in [24]:  # Bottom-right
            cv = feature[:, num]
            sv = torch.cat(
                (feature[:, num - 6].unsqueeze(1), feature[:, num - 5].unsqueeze(1), feature[:, num - 1].unsqueeze(1)),
                dim=1)
            return sv, cv


if __name__ == "__main__":
    Local_Aggregation = Local_Aggregation(channel=64, angRes=5).cuda()
    fea = torch.randn(1, 25, 64, 32, 32).cuda()
    output = Local_Aggregation(fea)
    print(output.shape)

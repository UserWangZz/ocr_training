import torch
import torch.nn as nn
from torch.nn import functional as F


class ASFBlock(nn.Module):
    def __init__(self, in_channels, inter_channels, out_features_num=4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super(ASFBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1)

        self.spatial_scale = nn.Sequential(
            # N*1*H*W
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=1
            ),
            nn.Sigmoid()
        )
        self.channel_scale = nn.Sequential(
            nn.Conv2d(
                in_channels=inter_channels,
                out_channels=out_features_num,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

    def forward(self, fuse_features, features_list):
        fuse_features = self.conv(fuse_features)
        spatial_x = torch.mean(fuse_features, dim=1, keepdim=True)
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        attention_scores = self.channel_scale(attention_scores)
        assert len(features_list) == self.out_features_num

        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[:, i:i + 1] * features_list[i])
            return torch.concat(out_list, dim=1)


class DBFPN(nn.Module):
    def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.use_asf = use_asf

        self.in2_conv = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels,
            kernel_size=1
        )
        self.in3_conv = nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels,
            kernel_size=1
        )
        self.in4_conv = nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels,
            kernel_size=1
        )
        self.in5_conv = nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels,
            kernel_size=1
        )

        self.p5_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1
        )
        self.p4_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1
        )
        self.p3_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1
        )
        self.p2_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1
        )
        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.interpolate(
            in5, scale_factor=2, mode='nearest')
        out3 = in3 + F.interpolate(
            in4, scale_factor=2, mode='nearest')
        out2 = in2 + F.interpolate(
            in3, scale_factor=2, mode='nearest')

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        p5 = F.interpolate(p5, scale_factor=8, mode='nearest')
        p4 = F.interpolate(p4, scale_factor=4, mode='nearest')
        p3 = F.interpolate(p3, scale_factor=2, mode='nearest')

        fuse = torch.concat([p5, p4, p3, p2], dim=1)
        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])

        return fuse

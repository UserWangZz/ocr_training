import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        out = out1 + out2
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  # 扩展系数，输出张量与输入张量通道数之比

    def __init__(self, in_channel, out_channel, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != self.expansion * out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, self.expansion * out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channel)
            )

    def forward(self, x):
        pic = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += self.shortcut(pic)
        out = F.relu(x)
        return out


class ResNet(nn.Module):
    out_channels = []

    def __init__(self, block, num_blocks, in_channels):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channels, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.__make_layer(block, 64, int(num_blocks[0]), stride=1)
        self.out_channels.append(64 * 4)
        self.layer2 = self.__make_layer(block, 128, int(num_blocks[1]), stride=2)
        self.out_channels.append(128 * 4)
        self.layer3 = self.__make_layer(block, 256, int(num_blocks[2]), stride=2)
        self.out_channels.append(256 * 4)
        self.layer4 = self.__make_layer(block, 512, int(num_blocks[3]), stride=2)
        self.out_channels.append(512 * 4)

    def __make_layer(self, block, out_channel, num_block, stride):
        # num_block控制Bottleneck串联个数
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for s in strides:
            if block == 'Bottleneck':
                layers.append(Bottleneck(self.in_channel, out_channel, s))
                self.in_channel = out_channel * Bottleneck.expansion
            else:
                layers.append(BasicBlock(self.in_channel, out_channel, s))
                self.in_channel = out_channel * BasicBlock.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.MaxPool(self.bn1(self.conv1(x))))
        out = []
        print(x.shape)
        x = self.layer1(x)
        # self.out_channels.append(x.shape[1])
        out.append(x)
        x = self.layer2(x)
        # self.out_channels.append(x.shape[1])
        out.append(x)
        x = self.layer3(x)
        # self.out_channels.append(x.shape[1])
        out.append(x)
        x = self.layer4(x)
        # self.out_channels.append(x.shape[1])
        out.append(x)

        return out

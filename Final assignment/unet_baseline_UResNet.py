# Original code and checkpoints by Hang Zhang
# https://github.com/zhanghang1989/PyTorch-Encoding


import math
import torch.nn as nn
import torch.nn.functional as F
import torch

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet50s-a75c83cf.zip',
    'resnet101': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet101s-03a0f310.zip',
    'resnet152': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet152s-36670e8b.zip'
}

class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResNet(nn.Module):
    """ResNet adapted for Semantic Segmentation with a fully convolutional output."""

    def __init__(self, block, layers, num_classes, dilated=True, multi_grid=False,
                 deep_base=True, norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()

        self.inplanes = 128 if deep_base else 64
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)

        self.upconv4 = self._upconv_block(512, 256)  # Upsample from 32x32 to 64x64
        self.upconv3 = self._upconv_block(256, 128)  # Upsample from 64x64 to 128x128
        self.upconv2 = self._upconv_block(128, 64)   # Upsample from 128x128 to 256x256

        # Final layer to output segmentation map
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # (64 + 64) channels from skip connections


        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            if multi_grid:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer, multi_grid=True)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        # Remove FC layer, replace with segmentation head
        self.segmentation_head = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _upconv_block(self, in_channels, out_channels):
        """Create an upconvolutional block with Conv2D and BatchNorm."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _decoder_block(self, in_channels, skip_channels, out_channels):
        """Decoder block with skip connection and refinement convolutions."""
        return nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass with improved decoder using skip connections."""
        # Encoder
        x1 = self.conv1(x)  # (batch, 128, 128, 128) or (batch, 64, 128, 128) depending on deep_base
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)  # (batch, 64, 64, 64)
        
        x2 = self.layer1(x1)  # (batch, 64, 64, 64)
        x3 = self.layer2(x2)  # (batch, 128, 32, 32)
        x4 = self.layer3(x3)  # (batch, 256, 32, 32)
        x5 = self.layer4(x4)  # (batch, 512, 32, 32)

        # Decoder with spatially matched skip connections
        # Upsample x5 and x4 to the same size for concatenation
        x = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)  # (batch, 512, 64, 64)
        x4_upsampled = F.interpolate(x4, size=x.shape[2:], mode='bilinear', align_corners=True)  # (batch, 256, 64, 64)

        # Apply decoder block - ensure it's on the same device as x
        decoder_block = self._decoder_block(512, x4.shape[1], 256).to(x.device)  # Move block to CUDA
        x = decoder_block(torch.cat([x, x4_upsampled], dim=1))  # (batch, 512, 64, 64)

        # Upsample x to the next size
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # (batch, 256, 128, 128)
        x3_upsampled = F.interpolate(x3, size=x.shape[2:], mode='bilinear', align_corners=True)  # (batch, 128, 128, 128)
        
        # Apply second decoder block - ensure it's on the same device as x
        x = self._decoder_block(256, 128, 128).to(x.device)(torch.cat([x, x3_upsampled], dim=1))  # (batch, 256, 128, 128)

        # Upsample x to the next size
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # (batch, 128, 256, 256)
        x2_upsampled = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=True)  # (batch, 64, 256, 256)
        
        # Apply third decoder block - ensure it's on the same device as x
        x = self._decoder_block(128, 64, 64).to(x.device)(torch.cat([x, x2_upsampled], dim=1))  # (batch, 128, 256, 256)

        # Final segmentation output
        x = self.final_conv(x)  # (batch, num_classes, 256, 256)

        return x




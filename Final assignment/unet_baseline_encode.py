import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # **Fix: Add a 1x1 convolution if in_channels != out_channels**
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust identity to match out_channels

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Now identity and out match in channels
        return self.relu(out)


class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super(Model, self).__init__()

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.layer1 = self._make_layer(64, 32, 4)
        
        # Stage 2
        self.transition1 = self._make_transition([64], [32, 64])  # Pass as lists
        self.stage2 = nn.ModuleList([
            self._make_layer(32, 32, 4),
            self._make_layer(64, 64, 4)
        ])
        
        # Stage 3
        self.transition2 = self._make_transition([32, 64], [32, 64, 128])  # Pass as lists
        self.stage3 = nn.ModuleList([
            self._make_layer(32, 32, 4),
            self._make_layer(64, 64, 4),
            self._make_layer(128, 128, 4)
        ])
        
        # Stage 4
        self.transition3 = self._make_transition([32, 64, 128], [32, 64, 128, 256])  # Pass as lists
        self.stage4 = nn.ModuleList([
            self._make_layer(32, 32, 4),
            self._make_layer(64, 64, 4),
            self._make_layer(128, 128, 4),
            self._make_layer(256, 256, 4)
        ])

        # Final classifier
        self.final_layer = nn.Conv2d(32, n_classes, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = [BasicBlock(in_channels, out_channels)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_transition(self, prev_channels_list, new_channels_list):
        layers = []
        for prev_channels, new_channels in zip(prev_channels_list, new_channels_list):
            layers.append(nn.Conv2d(prev_channels, new_channels, kernel_size=3, stride=1, padding=1, bias=False))
        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        
        x_list = [t(x) for t in self.transition1]
        x_list = [stage(x_list[i]) for i, stage in enumerate(self.stage2)]
        
        x_list = [t(x_list[-1]) for t in self.transition2] + x_list
        x_list = [stage(x_list[i]) for i, stage in enumerate(self.stage3)]
        
        x_list = [t(x_list[-1]) for t in self.transition3] + x_list
        x_list = [stage(x_list[i]) for i, stage in enumerate(self.stage4)]

        out = self.final_layer(x_list[0])  # Use the highest resolution branch
        return out



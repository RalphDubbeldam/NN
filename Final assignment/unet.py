import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),  
    )
    def forward(self, inputs):
        return self.layers(inputs)
    
class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            #nn.Sigmoid() 
    )
    def forward(self, inputs):
        return self.layers(inputs)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=1):        
        super().__init__()
        if upsample:
            self.upconv = nn.ConvTranspose2d(in_channels*2, in_channels*2, kernel_size=2, stride=2)
        else:
            self.upconv = nn.Identity()
        self.layers = DoubleConv(in_channels * 2, out_channels, out_channels)
        
    def forward(self, x, skip_connection):
 
        target_height = x.size(2)
        target_width = x.size(3)
        skip_interp = F.interpolate(
            skip_connection, size=(target_height, target_width), mode='bilinear', align_corners=False)
        
        concatenated = torch.cat([skip_interp,  x], dim=1)   

        concatenated = self.upconv(concatenated)
            
        output = self.layers(concatenated)
        return output
    
    
class Model(nn.Module):
    def __init__(self, input_features=3,num_classes=16, pretrained=True,
                layer1_features=32, layer2_features=16,
                layer3_features=24, layer4_features=40, layer5_features=80):
        super(Model, self).__init__()
        self.effnet = models.efficientnet_b0(pretrained=pretrained)

        self.num_classes = num_classes

#         # Layer feature sizes
        self.input_features = input_features
        self.layer1_features = layer1_features
        self.layer2_features = layer2_features
        self.layer3_features = layer3_features
        self.layer4_features = layer4_features
        self.layer5_features = layer5_features
#         Encoder layers
        self.encoder1 = nn.Sequential(*list(self.effnet.features.children())[0])  #out 32,112*112
        self.encoder2 = nn.Sequential(*list(self.effnet.features.children())[1])  #out 16,112*112
        self.encoder3 = nn.Sequential(*list(self.effnet.features.children())[2])  #out 24,56*56
        self.encoder4 = nn.Sequential(*list(self.effnet.features.children())[3])  #out 40,28*28
        self.encoder5 = nn.Sequential(*list(self.effnet.features.children())[4])  #out 40,28*28
        
        del self.effnet
        
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False

        # Bottleneck Layer
        self.bottleneck = DoubleConv(self.layer5_features, self.layer5_features, self.layer5_features)   
        
        # Decoder layers
        self.decoder1 = DecoderBlock(self.layer5_features, self.layer4_features)
        self.decoder2 = DecoderBlock(self.layer4_features, self.layer3_features)
        self.decoder3 = DecoderBlock(self.layer3_features, self.layer2_features)
        self.decoder4 = DecoderBlock(self.layer2_features, self.layer1_features, upsample=0)
        self.decoder5 = DecoderBlock(self.layer1_features, self.layer1_features)        
        
        # Final layer
        self.final_conv = FinalLayer(self.layer1_features, self.num_classes)
        
        
    def forward(self, x):
        # Encoder (contracting path)
        output1 = self.encoder1(x)
        output2 = self.encoder2(output1)
        output3 = self.encoder3(output2)
        output4 = self.encoder4(output3)
        output5 = self.encoder5(output4)
        
        # Bottleneck Layer
        bn = self.bottleneck(output5)
        up1 = self.decoder1(bn,  output5)
        up2 = self.decoder2(up1, output4)
        up3 = self.decoder3(up2, output3)
        up4 = self.decoder4(up3, output2)
        up5 = self.decoder5(up4, output1) 
        
        # Final convolution to produce segmentation mask
        res = self.final_conv(up5)

        return res
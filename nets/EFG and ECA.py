import torch.nn as nn
import torch
import torch.nn.functional as F
from .CCFTA import ChannelTransformer
import torch

class GateFusion(nn.Module):
    def __init__(self, in_planes):
        super(GateFusion, self).__init__()

        self.gate_1 = nn.Conv2d(in_planes * 4, 1, kernel_size=1, padding=0, bias=True)
        self.gate_2 = nn.Conv2d(in_planes * 4, 1, kernel_size=1, padding=0, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        try:
            cat_fea = torch.cat((x1, x2), dim=1)
        except RuntimeError as e:
            # If they cannot be concatenated, resize x2
            # print(f"Concatenation error: {e}. Resizing x2.")
            x2_resized = F.interpolate(x2, size=(224, 224), mode='bilinear', align_corners=False)
            cat_fea = torch.cat((x1, x2_resized), dim=1)

        # print(cat_fea.shape)
        att_vec_1 = self.gate_1(cat_fea)
        att_vec_2 = self.gate_2(cat_fea)

        att_vec_cat = torch.cat((att_vec_1, att_vec_2), dim=1)
        att_vec_soft = self.softmax(att_vec_cat)

        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]

        try:
            x_fusion = x1 * att_soft_1 + x2 * att_soft_2
        except RuntimeError as e:
            # If they cannot be concatenated, resize x2
            # print(f"Concatenation error: {e}. Resizing x2.")
            x2_resized2 = F.interpolate(x2, size=(224, 224), mode='bilinear', align_corners=False)
            x_fusion = x1 * att_soft_1 + x2_resized2 * att_soft_2

        return x_fusion








class SEAndProcessing(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super().__init__()

        # SE模块
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 上采样模块
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # 最大池化模块
        self.maxpool = nn.MaxPool2d(kernel_size=1)  # 1x1池化保持原始尺寸

    def se_forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs).view(b, c)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x).view(b, c, 1, 1)
        outputs = x * inputs
        return outputs

    def forward(self, x):
        # x 的形状为 (4, 64, 224, 224)
        x = self.se_forward(x)  # 经过SE模块
        x = self.upsample(x)  # 上采样
        x = self.maxpool(x)  # 最大池化
        return x


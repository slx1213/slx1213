import torch.nn as nn
import torch
import torch.nn.functional as F
from .CCFTA import ChannelTransformer
import torch


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


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


class DownBlock(nn.Module):  # 下采样
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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


class CCA(nn.Module):
    """
    CCA Block
    """

    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        # print(up.shape)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class EAtransUnet(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False, ratio=4):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels * 2)

        self.down1 = DownBlock(in_channels * 2, in_channels * 2, nb_Conv=2)

        self.low_fusion = GateFusion(in_channels)

        self.edge_fusion0 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU())
        self.edge_fusion1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU())
        self.edge_fusion2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU())
        self.edge_fusion3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU())
        self.edge_fusion4 = nn.Sequential(nn.Conv2d(in_channels * 2, 64, kernel_size=1, stride=1, padding=0, bias=True),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU())

        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=config.patch_sizes)
        self.up4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.up = SEAndProcessing(in_channels)
        self.up1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss

    def forward(self, x):
        x = x.float()
        x0 = self.inc(x)

        x2 = self.down1(x0)

        low_x = self.low_fusion(x0, x2)
        edge_out0 = self.edge_fusion1(low_x)
        edge_out1 = self.edge_fusion2(edge_out0)
        edge_out2 = self.edge_fusion3(edge_out1)
        x1 = self.edge_fusion4(edge_out2)

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x6 = self.up(x1)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x6)
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)  # if nusing BCEWithLogitsLoss or class>1
        if self.vis:  # visualize the attention maps
            return logits, att_weights
        else:
            return logits



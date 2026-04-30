import math
from model.smt import smt_t
from model.MobileNetV2 import mobilenet_v2
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

TRAIN_SIZE = 384

class FCIFNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.rgb_backbone = smt_t(pretrained)
        self.d_backbone = mobilenet_v2(pretrained)

        self.fc4 = FeatureCorrection_s2c(dim=512)
        self.fc3 = FeatureCorrection_s2c(dim=256)
        self.fc2 = FeatureCorrection_s2c(dim=128)
        self.fc1 = FeatureCorrection_s2c(dim=64)

        self.adjust_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.adjust_conv3 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.adjust_conv2 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.adjust_conv1 = nn.Conv2d(192, 64, kernel_size=3, padding=1)

        # Fuse
        self.LF4 = LF(infeature=512)
        self.LF3 = LF(infeature=256)
        self.LF2 = LF(infeature=128)
        self.LF1 = LF(infeature=64)

        # Pred
        self.dec3 = DEC(inc=512, outc=256)
        self.dec2 = DEC(inc=256, outc=128)
        self.dec1 = DEC(inc=128, outc=64)


        self.d_trans_4 = Trans(320, 512)
        self.d_trans_3 = Trans(96, 256)
        self.d_trans_2 = Trans(32, 128)
        self.d_trans_1 = Trans(24, 64)

        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        )



    def forward(self, x_rgb, x_d):
        _, (rgb_1, rgb_2, rgb_3, rgb_4) = self.rgb_backbone(x_rgb)

        _, d_1, d_2, d_3, d_4 = self.d_backbone(x_d)

        d_4 = self.d_trans_4(d_4)
        d_3 = self.d_trans_3(d_3)
        d_2 = self.d_trans_2(d_2)
        d_1 = self.d_trans_1(d_1)

        # 保存 FCM 前特征
        rgb_1_raw, rgb_2_raw, rgb_3_raw, rgb_4_raw = rgb_1, rgb_2, rgb_3, rgb_4
        d_1_raw, d_2_raw, d_3_raw, d_4_raw = d_1, d_2, d_3, d_4

        rgb_1, d_1 = self.fc1(rgb_1, d_1)
        rgb_2, d_2 = self.fc2(rgb_2, d_2)
        rgb_3, d_3 = self.fc3(rgb_3, d_3)
        rgb_4, d_4 = self.fc4(rgb_4, d_4)

        rgb_4_adjusted = self.adjust_conv4(rgb_4)
        rgb_3_adjusted = self.adjust_conv3(
            torch.cat([F.interpolate(rgb_4_adjusted, scale_factor=2, mode="bilinear", align_corners=True), rgb_3], dim=1))
        rgb_2_adjusted = self.adjust_conv2(
            torch.cat([F.interpolate(rgb_3_adjusted, scale_factor=2, mode="bilinear", align_corners=True), rgb_2], dim=1))
        rgb_1_adjusted = self.adjust_conv1(
            torch.cat([F.interpolate(rgb_2_adjusted, scale_factor=2, mode="bilinear", align_corners=True), rgb_1], dim=1))

        d_4_adjusted = self.adjust_conv4(d_4)
        d_3_adjusted = self.adjust_conv3(
            torch.cat([F.interpolate(d_4_adjusted, scale_factor=2, mode="bilinear", align_corners=True), d_3], dim=1))
        d_2_adjusted = self.adjust_conv2(
            torch.cat([F.interpolate(d_3_adjusted, scale_factor=2, mode="bilinear", align_corners=True), d_2], dim=1))
        d_1_adjusted = self.adjust_conv1(
            torch.cat([F.interpolate(d_2_adjusted, scale_factor=2, mode="bilinear", align_corners=True), d_1], dim=1))

        fuse_4 = self.LF4(rgb_4_adjusted, d_4_adjusted)
        fuse_3 = self.LF3(rgb_3_adjusted, d_3_adjusted)
        fuse_2 = self.LF2(rgb_2_adjusted, d_2_adjusted)
        fuse_1 = self.LF1(rgb_1_adjusted, d_1_adjusted)


        pred_4 = F.interpolate(self.predtrans(fuse_4), TRAIN_SIZE, mode="bilinear", align_corners=True)
        pred_3, xf_3 = self.dec3(fuse_3, fuse_4)
        pred_2, xf_2 = self.dec2(fuse_2, xf_3)
        pred_1, xf_1 = self.dec1(fuse_1, xf_2)

        return pred_1, pred_2, pred_3, pred_4, {
            'rgb_2_raw': rgb_2_raw,
            'd_2_raw': d_2_raw,
            'rgb_2': rgb_2,
            'd_2': d_2,
            'fuse_2': fuse_2,
            'rgb_3_raw': rgb_3_raw,
            'd_3_raw': d_3_raw,
            'rgb_3': rgb_3,
            'd_3': d_3,
            'fuse_3': fuse_3
        }


class Trans(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.apply(self._init_weights)

    def forward(self, d):
        return self.trans(d)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class DEC(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, groups=outc),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(in_channels=outc, out_channels=1, kernel_size=1)
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=outc * 2, out_channels=outc * 2, kernel_size=3, padding=1, groups=outc * 2),
            nn.BatchNorm2d(outc * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=outc * 2, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

        self.rc3 = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, stride=1, groups=outc),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

        self.rc4 = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, stride=1, groups=outc),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x2_upsample = self.upsample2(x2)

        x2_rc = self.rc(x2_upsample)

        x_cat = torch.cat((x1, x2_rc), dim=1)
        x_cat_rc = self.rc2(x_cat)

        x_mul = x1 * x2_rc
        x_mul_rc = self.rc3(x_mul)

        x_combined = x_cat_rc + x_mul_rc

        x_forward = self.rc4(x_combined)

        pred = F.interpolate(self.predtrans(x_forward), TRAIN_SIZE, mode="bilinear", align_corners=True)

        return pred, x_forward



class LF(nn.Module):
    def __init__(self, infeature):
        super(LF, self).__init__()
        self.lga_p2 = LocalGlobalAttention(infeature, patch_size=2)
        self.lga_p4 = LocalGlobalAttention(infeature, patch_size=4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(infeature, infeature, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(infeature),
            nn.GELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(infeature, infeature, kernel_size=1, stride=1),
            nn.BatchNorm2d(infeature),
            nn.GELU(),
        )

    def forward(self, r, d):
        x_add = r + d

        x_p2 = self.lga_p2(x_add)

        x_p2_add = x_p2 * r + x_p2 * d

        x_p2_add = self.conv1(x_p2_add)

        x_p4 = self.lga_p4(x_p2_add)

        p4_r = x_p4 * r

        p4_d = x_p4 * d

        F = p4_r + p4_d

        F = self.conv2(F)

        return F

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size
        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)
        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)
        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)
        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform
        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)
        return output


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())
    def forward(self, x1, x2):
        B, _, H, W = x1.shape

        avg1 = self.avg_pool(x1).view(B, self.dim)
        avg2 = self.avg_pool(x2).view(B, self.dim)

        max1 = self.max_pool(x1).view(B, self.dim)
        max2 = self.max_pool(x2).view(B, self.dim)

        y = torch.cat((avg1, avg2, max1, max2), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights

class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights

class FeatureCorrection_s2c(nn.Module):
    def __init__(self, dim, reduction=1, eps=1e-8):
        super(FeatureCorrection_s2c, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        spatial_weights = self.spatial_weights(x1, x2)
        x1_1 = x1 + fuse_weights[0] * spatial_weights[1] * x2
        x2_1 = x2 + fuse_weights[0] * spatial_weights[0] * x1

        channel_weights = self.channel_weights(x1_1, x2_1)

        r_out = x1_1 + fuse_weights[1] * channel_weights[1] * x2_1
        d_out = x2_1 + fuse_weights[1] * channel_weights[0] * x1_1
        return r_out, d_out



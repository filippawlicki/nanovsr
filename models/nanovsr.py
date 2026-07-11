import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        if self.deploy:
            return self.activation(self.rbr_reparam(x))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)

        return self.activation(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)

        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense[0].in_channels,
            out_channels=self.rbr_dense[0].out_channels,
            kernel_size=self.rbr_dense[0].kernel_size,
            stride=self.rbr_dense[0].stride,
            padding=self.rbr_dense[0].padding,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        self.deploy = True


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.prelu(x)


class NanoVSR(nn.Module):
    def __init__(self, num_feat=32, num_blocks=7, deploy=False):
        super(NanoVSR, self).__init__()
        self.num_feat = num_feat
        self.deploy = deploy

        self.feat_extract = RepVGGBlock(3, num_feat, deploy=deploy)

        self.forward_net = nn.Sequential(*[RepVGGBlock(num_feat, num_feat, deploy=deploy) for _ in range(num_blocks)])
        self.backward_net = nn.Sequential(*[RepVGGBlock(num_feat, num_feat, deploy=deploy) for _ in range(num_blocks)])

        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)

        self.upsample1 = PixelShuffleBlock(num_feat, num_feat, upscale_factor=2)
        self.upsample2 = PixelShuffleBlock(num_feat, 32, upscale_factor=2)

        self.conv_last = nn.Conv2d(32, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        b, t, c, h, w = x.size()

        x_flat = x.view(-1, c, h, w)
        feats = self.feat_extract(x_flat)
        feats = feats.view(b, t, -1, h, w)

        forward_feats = []
        feat_prop = torch.zeros_like(feats[:, 0, ...])
        for i in range(t):
            feat_prop = self.forward_net(feats[:, i, ...] + feat_prop)
            forward_feats.append(feat_prop)

        backward_feats = []
        feat_prop = torch.zeros_like(feats[:, 0, ...])
        for i in range(t - 1, -1, -1):
            feat_prop = self.backward_net(feats[:, i, ...] + feat_prop)
            backward_feats.insert(0, feat_prop)

        outputs = []

        for i in range(t):
            f_fused = torch.cat([forward_feats[i], backward_feats[i]], dim=1)
            f_fused = self.fusion(f_fused)

            out = self.upsample1(f_fused)
            out = self.upsample2(out)
            out = self.conv_last(out)

            base = F.interpolate(x[:, i, ...], scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            outputs.append(out)

        final_video = torch.stack(outputs, dim=1)

        return final_video

    def switch_to_deploy(self):
        if self.deploy:
            return

        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        self.deploy = True
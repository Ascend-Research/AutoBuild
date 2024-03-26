import torch
import numpy as np
import torch.nn as nn
from search.rm_search.ofa_profile.arch_utils import get_final_channel_size


BN_MOMENTUM = 0.1
BN_EPSILON = 1e-3


def _decode_kernel_padding_size(op):
    k = [int(s) for s in op.split("x")]
    assert all(v % 2 != 0 for v in k), "Only odd kernel sizes supported"
    padding = [v // 2 for v in k]
    return k, padding


def _decode_mbconv_e_k(op):
    _op = op.replace("mbconv", "")
    args = _op.split("_")[1:]
    e, k = args
    e = int(e.replace("e", ""))
    k = int(k.replace("k", ""))
    return e, k


def get_torch_op(C_in, C_out, stride, affine, op_name,
                 act_type="relu", se_ratio=0.):
    if op_name.startswith("mbconv2"):
        e, k = _decode_mbconv_e_k(op_name)
        return MBConv2(C_in, C_out, k, stride,
                       affine=affine, expansion_ratio=e,
                       enable_skip=stride==1 and C_in==C_out)
    elif op_name.startswith("mbconv3"):
        e, k = _decode_mbconv_e_k(op_name)
        return MBConv3(C_in, C_out, k, stride,
                       se_ratio=se_ratio, affine=affine, expansion_ratio=e,
                       enable_skip=stride==1 and C_in==C_out,
                       act_type=act_type)
    else:
        raise ValueError("Unknown op name: {}".format(op_name))


class ConvBNReLU6(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvBNReLU6, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM, eps=BN_EPSILON),
            nn.ReLU6(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


class ConvBNReLU(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvBNReLU, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM, eps=BN_EPSILON),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


class ConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM, eps=BN_EPSILON)
        )

    def forward(self, x):
        return self.op(x)


class MBConv2(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, affine=True,
                 expansion_ratio=6, enable_skip=True):
        super(MBConv2, self).__init__()
        self.enable_skip = enable_skip and stride == 1 and C_in == C_out
        self.expanded_channel_size = int(C_in * expansion_ratio)
        self.expansion_conv = None
        if expansion_ratio > 1:
            self.expansion_conv = ConvBNReLU6(C_in, self.expanded_channel_size,
                                              kernel_size=1, stride=1, padding=0, affine=affine)
        self.depth_conv =  nn.Sequential(nn.Conv2d(self.expanded_channel_size, self.expanded_channel_size,
                                                   kernel_size=kernel_size, stride=stride,
                                                   padding=kernel_size // 2,
                                                   groups=self.expanded_channel_size, bias=False),
                                         nn.BatchNorm2d(self.expanded_channel_size,
                                                        affine=affine, momentum=BN_MOMENTUM, eps=BN_EPSILON),
                                         nn.ReLU6(inplace=False))
        self.point_conv = ConvBN(self.expanded_channel_size, C_out,
                                 kernel_size=1, stride=1, padding=0, affine=affine)

    def forward(self, inputs):
        if self.expansion_conv is not None:
            x = self.expansion_conv(inputs)
        else:
            x = inputs
        x = self.depth_conv(x)
        x = self.point_conv(x)
        if self.enable_skip:
            x += inputs
        return x


class ConvBNSwish(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvBNSwish, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM, eps=BN_EPSILON),
            HSwish()
        )

    def forward(self, x):
        return self.op(x)


class MBConv3(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, affine=True,
                 expansion_ratio=6, se_ratio=0.25, enable_skip=True,
                 act_type="relu"):
        super(MBConv3, self).__init__()
        self.enable_skip = enable_skip and stride == 1 and C_in == C_out
        self.expanded_channel_size = int(C_in * expansion_ratio)
        self.has_se = 1 >= se_ratio > 0
        self.se_channel_size = get_final_channel_size(max(1, int(self.expanded_channel_size * se_ratio)), 1.0)
        self.expansion_conv = None
        self.act_type = act_type
        if expansion_ratio > 1:
            if self.act_type == "swish":
                self.expansion_conv = ConvBNSwish(C_in, self.expanded_channel_size,
                                                 kernel_size=1, stride=1, padding=0, affine=affine)
            elif self.act_type == "relu":
                self.expansion_conv = ConvBNReLU(C_in, self.expanded_channel_size,
                                                 kernel_size=1, stride=1, padding=0, affine=affine)
            else:
                raise ValueError("Unknown act type: {}".format(self.act_type))
        self.depth_conv =  nn.Sequential(nn.Conv2d(self.expanded_channel_size, self.expanded_channel_size,
                                                   kernel_size=kernel_size, stride=stride,
                                                   padding=kernel_size // 2,
                                                   groups=self.expanded_channel_size, bias=False),
                                         nn.BatchNorm2d(self.expanded_channel_size,
                                                        affine=affine, momentum=BN_MOMENTUM, eps=BN_EPSILON),
                                         HSwish() if self.act_type=="swish" else torch.nn.ReLU())
        self.se_module = None
        if self.has_se:
            self.se_module = SqueezeExcite(self.expanded_channel_size, self.se_channel_size)
        self.point_conv = ConvBN(self.expanded_channel_size, C_out,
                                 kernel_size=1, stride=1, padding=0, affine=affine)

    def forward(self, inputs):
        if self.expansion_conv is not None:
            x = self.expansion_conv(inputs)
        else:
            x = inputs
        x = self.depth_conv(x)
        if self.has_se:
            x = self.se_module(x)
        x = self.point_conv(x)
        if self.enable_skip:
            x += inputs
        return x


class SqueezeExcite(nn.Module):

    def __init__(self, C_in, C_squeeze):
        super(SqueezeExcite, self).__init__()
        self.squeeze_conv = nn.Conv2d(C_in, C_squeeze, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.excite_conv = nn.Conv2d(C_squeeze, C_in, kernel_size=1, stride=1, padding=0, bias=True)
        self.h_sigmoid = HSigmoid()

    def forward(self, x):
        x_sq = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x_sq = self.squeeze_conv(x_sq)
        x_sq = self.relu(x_sq)
        x_sq = self.excite_conv(x_sq)
        return self.h_sigmoid(x_sq) * x


class HSigmoid(nn.Module):

    def __init__(self):
        super(HSigmoid, self).__init__()

    def forward(self, x):
        return torch.nn.functional.relu6(x + 3.) / 6.


class HSwish(nn.Module):

    def __init__(self):
        super(HSwish, self).__init__()

    def forward(self, x):
        return x * torch.nn.functional.relu6(x + np.float32(3)) * (1. / 6.)

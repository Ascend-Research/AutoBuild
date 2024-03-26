from search.rm_search.ofa_profile.constants import *
from search.rm_search.ofa_profile.operations import *
from search.rm_search.ofa_profile.arch_utils import get_final_channel_sizes


class OFAProxylessStem(torch.nn.Module):

    def __init__(self, C_in, C_out=32):
        super(OFAProxylessStem, self).__init__()
        self.conv = torch.nn.Conv2d(C_in, out_channels=C_out, kernel_size=3,
                                    stride=2, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(C_out, affine=True, momentum=BN_MOMENTUM, eps=BN_EPSILON)
        self.act = torch.nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class OFAMbv3Stem(torch.nn.Module):

    def __init__(self, C_in, C_out=16):
        super(OFAMbv3Stem, self).__init__()
        self.conv = torch.nn.Conv2d(C_in, out_channels=C_out, kernel_size=3,
                                    stride=2, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(C_out, affine=True)
        self.act = HSwish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class OFAProxylessOutLayer(torch.nn.Module):

    def __init__(self, C_in, C_hidden, n_classes):
        super(OFAProxylessOutLayer, self).__init__()
        self.conv = torch.nn.Conv2d(C_in, out_channels=C_hidden, kernel_size=1,
                                    stride=1, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(C_hidden, affine=True, momentum=BN_MOMENTUM, eps=BN_EPSILON)
        self.act = torch.nn.ReLU6()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_hidden, n_classes)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        out = self.global_pooling(x)
        out = torch.dropout(out, 0.1, train=self.training)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class OFAMbv3OutLayer(torch.nn.Module):

    def __init__(self, C_in, hidden_1, hidden_2, n_classes):
        super(OFAMbv3OutLayer, self).__init__()
        self.conv1 = torch.nn.Conv2d(C_in, out_channels=hidden_1, kernel_size=1,
                                     stride=1, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(hidden_1, affine=True)
        self.act = HSwish()
        self.conv2 = torch.nn.Conv2d(hidden_1, out_channels=hidden_2, kernel_size=1,
                                     stride=1, padding=0, bias=False)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(hidden_2, n_classes)

    def forward(self, x):
        x = self.act(self.bn(self.conv1(x)))
        x = self.global_pooling(x)
        out = self.act(self.conv2(x))
        out = out.reshape(out.size(0), out.size(1))
        out = torch.dropout(out, 0.1, train=self.training)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class OFAProxylessNet(torch.nn.Module):

    def __init__(self, net_configs,  # A list of block name lists, group by stages
                 C_input=3, C_init=32, n_classes=1000,
                 strides=(2, 2, 2, 1, 2, 1),
                 block_channel_sizes=(16, 24, 40, 80, 96, 192, 320),
                 out_hidden_size=1280, w=OFA_W_PN):
        super(OFAProxylessNet, self).__init__()
        assert len(net_configs) == len(strides) == len(block_channel_sizes) - 1
        C_init = get_final_channel_size(C_init, w)
        block_channel_sizes = get_final_channel_sizes(block_channel_sizes, w)
        out_hidden_size = get_final_channel_size(out_hidden_size, w)
        self.w = w
        self.net_configs = net_configs
        self.strides = strides
        self.block_channel_sizes = block_channel_sizes
        self.stem = OFAProxylessStem(C_input, C_init)
        first_block = MBConv2(C_init, block_channel_sizes[0], kernel_size=3, stride=1,
                              affine=True, expansion_ratio=1, enable_skip=False)
        self.blocks = nn.ModuleList([first_block])
        C_in = block_channel_sizes[0]
        for i, b_ops in enumerate(self.net_configs):
            b_stride = strides[i]
            C_out = self.block_channel_sizes[i + 1]
            for op_i, op_name in enumerate(b_ops):
                block = get_torch_op(C_in=C_in, C_out=C_out,
                                     stride=b_stride if op_i == 0 else 1,
                                     affine=True, op_name=op_name)
                self.blocks.append(block)
                C_in = C_out
        self.out_net = OFAProxylessOutLayer(C_in, out_hidden_size, n_classes)

    def forward(self, x):
        x = self.stem(x)
        for bi, block in enumerate(self.blocks):
            x = block(x)
        return self.out_net(x)


class OFAMbv3Net(torch.nn.Module):

    def __init__(self, net_configs,  # A list of block name lists, group by stages
                 C_input=3, C_init=16, n_classes=1000,
                 strides=(2, 2, 2, 1, 2),
                 block_channel_sizes=(16, 24, 40, 80, 112, 160),
                 stage_acts=("relu", "relu", "swish", "swish", "swish"),
                 stage_se_ratios=(0., 0.25, 0., 0.25, 0.25),
                 hidden_1=960, hidden_2=1280, w=OFA_W_MBV3):
        super(OFAMbv3Net, self).__init__()
        assert len(net_configs) == len(strides) == len(block_channel_sizes) - 1
        C_init = get_final_channel_size(C_init, w)
        block_channel_sizes = get_final_channel_sizes(block_channel_sizes, w)
        hidden_1 = get_final_channel_size(hidden_1, w)
        hidden_2 = get_final_channel_size(hidden_2, w)
        self.net_configs = net_configs
        self.strides = strides
        self.block_channel_sizes = block_channel_sizes
        self.stem = OFAMbv3Stem(C_input, C_init)
        first_block = MBConv3(C_init, block_channel_sizes[0], kernel_size=3, stride=1,
                              affine=True, expansion_ratio=1, enable_skip=True, se_ratio=0.,
                              act_type="relu")
        self.blocks = nn.ModuleList([first_block])
        C_in = block_channel_sizes[0]
        for i, b_ops in enumerate(self.net_configs):
            b_stride = strides[i]
            C_out = self.block_channel_sizes[i + 1]
            act_type = stage_acts[i]
            se_ratio = stage_se_ratios[i]
            for op_i, op_name in enumerate(b_ops):
                block = get_torch_op(C_in=C_in, C_out=C_out,
                                     stride=b_stride if op_i == 0 else 1,
                                     affine=True, op_name=op_name,
                                     act_type=act_type, se_ratio=se_ratio)
                self.blocks.append(block)
                C_in = C_out
        self.out_net = OFAMbv3OutLayer(C_in, hidden_1, hidden_2, n_classes)

    def forward(self, x):
        x = self.stem(x)
        for bi, block in enumerate(self.blocks):
            x = block(x)
        return self.out_net(x)

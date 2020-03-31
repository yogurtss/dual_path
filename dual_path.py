import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import *


class LowHighNetwork(nn.Module):
    def __init__(self, in_chs, low_in_chs, low_out_chs, high_in_chs,
                 high_out_chs, l_low=24, l_high=24//4, k=3, number_cls=2, number_block=4):
        super(LowHighNetwork, self).__init__()
        self.number_block = number_block
        self.number_cls = number_cls
        self.encoder_low = Encoder(in_chs, low_in_chs, l_low)
        self.encoder_high = Encoder(in_chs, high_in_chs, l_high)
        for i in range(number_block):
            if i == 0:
                block = LowHighBlock(low_in_chs, low_out_chs, high_in_chs, high_out_chs, k)
            else:
                block = LowHighBlock(low_in_chs+high_in_chs, low_out_chs, high_in_chs, high_out_chs, k, is_start=False)
            setattr(self, 'block_{}'.format(i), block)
        self.conv_fusion = nn.Conv1d(low_in_chs+high_in_chs, low_in_chs, 3, 1, 1)
        self.mask = nn.Conv1d(low_in_chs, number_cls * low_in_chs, 1, 1)
        self.decoder = Decoder(low_in_chs, l_low)

    def forward(self, x):
        low = self.encoder_low(x)
        high = self.encoder_high(x)
        x = [low, high]
        for i in range(self.number_block):
            block = getattr(self, 'block_{}'.format(i))
            x = block(x)
        x = self.conv_fusion(x[0])
        M, N, K = x.size()
        x = self.mask(x)
        x = x.view(M, self.number_cls, N, K)
        x = F.relu(x)
        x = self.decoder(low, x)
        x = x.clamp(-1.0, 1.0)
        return x


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, in_chs, out_chs, l, stride=0):
        '''

        :param in_chs: number of the input channel
        :param out_chs: number of the output channel
        :param l: length of filter
        '''
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.in_chs, self.out_chs, self.l = in_chs, out_chs, l
        if stride == 0:
            self.stride = l // 2
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(in_chs, out_chs, kernel_size=l, stride=self.stride, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, n_chs, l):
        super(Decoder, self).__init__()
        self.n_chs, self.l = n_chs, l
        self.convert = nn.Linear(n_chs, l, bias=False)

    def forward(self, mixture, mask):
        '''
        Args:
            mixture: mixture [M, N, K]
            mask: [M, C, N, K]
        '''
        source_w = mixture.unsqueeze(1) * mask      # [M, C, N, K]
        source_w = torch.transpose(source_w, 2, 3)  # [M, C, K, N]
        est_source = self.convert(source_w)
        est_source = overlap_and_add(est_source, self.l//2)
        return est_source


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)
        prelu = nn.PReLU()
        norm = nn.GroupNorm(in_channels//8, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, kernel_size,
                 stride, padding, dilation):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_chs, mid_chs, 1, bias=False)
        prelu = nn.PReLU()
        norm = nn.GroupNorm(mid_chs//8, mid_chs)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(mid_chs, out_chs, kernel_size,
                                        stride, padding, dilation)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        return self.net(x)


class LowHighBlock(nn.Module):
    def __init__(self, low_in_chs, low_out_chs, high_in_chs, high_out_chs,
                 kernel_size, n_block=4, is_padding=True, is_start=True):
        super(LowHighBlock, self).__init__()
        padding = 0
        self.n_block = n_block
        low = []
        high = []
        if not is_start:
            low_final_chs = low_in_chs - high_in_chs
        else:
            low_final_chs = low_in_chs
        for i in range(n_block):
            dilation = 2**i
            if is_padding:
                padding = (kernel_size - 1) * dilation // 2
                low += [TemporalBlock(low_in_chs, low_out_chs, low_final_chs, kernel_size,
                                      stride=1, padding=padding, dilation=dilation)]
            low_in_chs = low_final_chs
        self.low = nn.Sequential(*low)

        for i in range(n_block):
            dilation = 2 ** i
            if is_padding:
                padding = (kernel_size - 1) * dilation // 2
            high += [TemporalBlock(high_in_chs, high_out_chs, high_in_chs, kernel_size,
                                   stride=1, padding=padding, dilation=dilation)]
        self.high = nn.Sequential(*high)
        self.lat = nn.Conv1d(high_in_chs, high_in_chs, kernel_size=7, stride=4)

    def forward(self, x):
        low, high = x
        high = self.high(high)
        low = self.low(low)
        low = torch.cat([low, self.lat(high)], dim=1)
        return [low, high]


if __name__ == '__main__':
    a = LowHighNetwork(1, 256, 512, 32, 64)
    print(a)
    b = torch.zeros([1, 1, 31992])
    c = a(b)
    print('test')



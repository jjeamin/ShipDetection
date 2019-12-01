import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        return x


class SamePadConv2d(nn.Conv2d):
    """
    Conv with TF padding='same'
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_)
        rows_odd = (padding_rows % 2 != 0)
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0])
        padding_cols, cols_odd = self.get_pad_odd(x.shape[3], self.weight.shape[3], self.stride[1], self.dilation[1])

        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(x, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)


class Conv_bn_relu(nn.Module):
    def __init__(self, in_planes, planes,
                 kernel_size, stride=1, padding=0, bias=True, leaky=False):
        super(Conv_bn_relu, self).__init__()

        if padding == 'SAME':
            self.conv = SamePadConv2d(in_planes, planes,
                                      stride=stride, kernel_size=kernel_size, bias=bias)

        else:
            self.conv = nn.Conv2d(in_planes, planes,
                                  stride=stride, kernel_size=kernel_size,
                                  padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(planes)

        if leaky:
            self.relu = nn.LeakyReLU(0.1)
        else:
            self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv_bn(nn.Module):
    def __init__(self, in_planes, planes,
                 kernel_size, stride=1, padding=0, bias=True):

        super(Conv_bn, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return self.bn(self.conv(x))

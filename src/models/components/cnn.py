import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


def conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
               use_BN: bool = True):
    """
    Simple convolutional block

    :param in_channels: int
        number of input channels
    :param out_channels: int
        number of output channels
    :param kernel_size: int
        kernel size
    :param stride: int
        Stride of the convolution
    :param padding: int
        padded value
    :param use_BN: bool
        if BN is applied
    :return: conv_block: torch.nn.Module
        a convolutional block layer
    """
    c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    a = nn.ReLU(inplace=False)
    if use_BN:
        b = nn.BatchNorm2d(out_channels)
        return nn.Sequential(c, a, b)
    else:
        return nn.Sequential(c, a)


class CNN(nn.Module):
    """
    The model to optimize
    """

    def __init__(
        self,
        input_shape=(1, 28, 28), 
        n_conv_layers: int = 3,
        kernel_size: int = 5,
        batch_norm: bool = False,
        global_avg_pooling: bool = True,
        in_channels: int = 1,
        key_conv: str = 'n_channels_conv_',
        key_fc: str = 'n_channels_fc_',
        out_channels: int = 16,
        dropout_rate: float = 0.2,
        n_fc_layers: int = 3,
        num_classes: int = 10
    ):
        super(CNN, self).__init__()
        layers = []

        for i in range(n_conv_layers):
            padding = (kernel_size - 1) // 2
            conv_block_0 = conv_block(in_channels, out_channels, kernel_size=kernel_size,
                                      padding=padding, use_BN=batch_norm)
            p = nn.MaxPool2d(kernel_size=2, stride=2)
            layers.extend([conv_block_0, p])
            in_channels = out_channels
            out_channels_tmp = key_conv + str((i + 1))
            out_channels = out_channels_tmp if out_channels_tmp else out_channels * 2

        self.conv_layers = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool2d(1) if global_avg_pooling else nn.Identity()
        self.output_size = num_classes

        self.fc_layers = nn.ModuleList()
        n_in = self._get_conv_output(input_shape)
        key_fc = 'n_channels_fc_'
        n_out = key_fc + '0'
        n_out = n_out if n_out else 256

        
        config = {'n_channels_fc_0': 27,
                    'n_channels_fc_1': 17,
                    'n_channels_fc_2': 273}
        for i in range(n_fc_layers):
            fc = nn.Linear(int(n_in), int(n_out))
            self.fc_layers += [fc]
            n_in = n_out
            n_out_tmp = key_fc + str((i + 1))
            n_out = n_out_tmp if n_out_tmp else n_out / 2

        self.last_fc = nn.Linear(int(n_in), self.output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.time_train = 0

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        output_feat = self.pooling(output_feat)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = self.dropout(F.relu(fc_layer(x)))
        x = self.last_fc(x)
        return x
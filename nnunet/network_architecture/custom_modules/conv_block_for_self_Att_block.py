#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from copy import deepcopy
from nnunet.network_architecture.custom_modules.helperModules import Identity
from torch import nn


class ConvDropoutNormReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props):
        """
        if network_props['dropout_op'] is None then no dropout
        if network_props['norm_op'] is None then no norm
        :param input_channels:
        :param output_channels:
        :param kernel_size:
        :param network_props:
        """
        super(ConvDropoutNormReLU, self).__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.conv = network_props['conv_op'](input_channels, output_channels, kernel_size,
                                             padding=[(i - 1) // 2 for i in kernel_size],
                                             **network_props['conv_op_kwargs'])

        # maybe dropout
        if network_props['dropout_op'] is not None:
            self.do = network_props['dropout_op'](**network_props['dropout_op_kwargs'])
        else:
            self.do = Identity()

        if network_props['norm_op'] is not None:
            self.norm = network_props['norm_op'](output_channels, **network_props['norm_op_kwargs'])
        else:
            self.norm = Identity()

        self.nonlin = network_props['nonlin'](**network_props['nonlin_kwargs'])

        self.all = nn.Sequential(self.conv, self.do, self.norm, self.nonlin)

    def forward(self, x):
        return self.all(x)


class StackedConvLayers(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_convs, first_stride=None):
        """
        if network_props['dropout_op'] is None then no dropout
        if network_props['norm_op'] is None then no norm
        :param input_channels:
        :param output_channels:
        :param kernel_size:
        :param network_props:
        """
        super(StackedConvLayers, self).__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.
        network_props_first = deepcopy(network_props)

        if first_stride is not None:
            network_props_first['conv_op_kwargs']['stride'] = first_stride

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(input_channels, output_channels, kernel_size, network_props_first),
            *[ConvDropoutNormReLU(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_convs - 1)]
        )

    def forward(self, x):
        return self.convs(x)





class Attention_block(nn.Module):
    def __init__(self, F_x, F_y,F_z, F_int):
        super(Attention_block, self).__init__()


        self.W_x = nn.Sequential(
            nn.Conv3d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_y = nn.Sequential(
            nn.Conv3d(F_y, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_z = nn.Sequential(
            nn.Conv3d(F_z, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            # nn.Sigmoid()
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, g, x, y, z):

        x1 = self.W_x(x)
        y1 = self.W_y(y)
        z1 = self.W_z(z)
        sum = x1 + y1+ z1
        # sum = sum * 1.5

        psi = self.relu(sum)
        psi = self.psi(psi)
        # psi = psi * 1.5

        return g * psi, psi

# sigmoid를 LeakyReLU로도 바꿔보고
# sum = 1.5로도 바꿔보면서 학습해보자





class DenseDownBlock_first(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        # if props['dropout_op_kwargs'] is None and props['dropout_op_kwargs'] > 0:
        #     raise NotImplementedError("ResidualBottleneckBlock does not yet support dropout!")

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.bottleneck_planes = out_planes // 4

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = (1,1,1)
        else:
            kwargs_conv1 = props['conv_op_kwargs']

        self.pool_op = nn.MaxPool3d(2,stride=2)


        # small version
        self.conv1 = props['conv_op'](in_planes, in_planes, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props['norm_op'](in_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv2 = props['conv_op'](in_planes * 2, in_planes, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](in_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])




    def forward(self, x):


        # small version
        residual_1 = x  # 32

        out_1 = self.nonlin1(self.norm1(self.conv1(x)))  # 32
        residual_2 = out_1
        concat_1 = torch.cat((out_1, residual_1), dim=1)  # 32 * 2

        residual_out = self.nonlin2(self.norm2(self.conv2(concat_1)))  # 32 * 2

        concat_2 = torch.cat((concat_1, residual_1), dim=1)


        out = self.pool_op(residual_out)

        return out , residual_out, residual_1, concat_1, concat_2








# 여기서 DenseBlock 구현

class DenseDownBlock_2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        # if props['dropout_op_kwargs'] is None and props['dropout_op_kwargs'] > 0:
        #     raise NotImplementedError("ResidualBottleneckBlock does not yet support dropout!")

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.bottleneck_planes = out_planes // 4

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = (1,1,1)
        else:
            kwargs_conv1 = props['conv_op_kwargs']


        # maxpooling 구현
        self.pool_op = nn.MaxPool3d(2,stride=2)


        self.conv1 = props['conv_op'](in_planes, in_planes, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props['norm_op'](in_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv2 = props['conv_op'](in_planes * 2, in_planes*2, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](in_planes*2, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv3 = props['conv_op'](in_planes * 4, in_planes * 2, [1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm3 = props['norm_op'](in_planes * 2, **props['norm_op_kwargs'])
        self.nonlin3 = props['nonlin'](**props['nonlin_kwargs'])



    def forward(self, x):


        residual_1 = x  # 32

        out_1 = self.nonlin1(self.norm1(self.conv1(x)))  # 32
        residual_2 = out_1
        concat_1 = torch.cat((out_1, residual_1), dim=1)  # 32 * 2

        out = self.nonlin2(self.norm2(self.conv2(concat_1)))  # 32 * 2



        concat_2 = torch.cat((out, residual_1), dim=1)  # 32*2 + 32*1 = 32 * 3
        concat_2 = torch.cat((concat_2,residual_2), dim = 1) # 32*3 + 32* = 32 * 4


        residual_out = self.nonlin3(self.norm3(self.conv3(concat_2)))
        out = self.pool_op(residual_out)




        return out ,residual_out, residual_1, concat_1, concat_2





class DenseDownLayer_2(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None, block=DenseDownBlock_2):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            block(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[block(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)






class DenseDownLayer_first(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None, block=DenseDownBlock_first):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            block(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[block(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)











# 여기는 Dense_Up_Block 구현하기



class DenseUpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        # if props['dropout_op_kwargs'] is None and props['dropout_op_kwargs'] > 0:
        #     raise NotImplementedError("ResidualBottleneckBlock does not yet support dropout!")

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes
        # self.bottleneck_planes = out_planes // 4

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = (1,1,1)
        else:
            kwargs_conv1 = props['conv_op_kwargs']







        # small version

        aim_planes = in_planes // 2  # 256

        self.conv0 = props['conv_op'](in_planes, aim_planes, [1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm0 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin0 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv1 = props['conv_op'](aim_planes, aim_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv2 = props['conv_op'](aim_planes * 2, aim_planes, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv3 = props['conv_op'](aim_planes * 3, aim_planes, [1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm3 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin3 = props['nonlin'](**props['nonlin_kwargs'])








    def forward(self, x):


        x = self.nonlin0(self.norm0(self.conv0(x)))  # 256
        residual_1 = x  # 256

        out_1 = self.nonlin1(self.norm1(self.conv1(x)))  # 256
        residual_2 = out_1  # 256
        concat_1 = torch.cat((out_1, residual_1), dim=1)  # 512

        out = self.nonlin2(self.norm2(self.conv2(concat_1)))  # 256


        concat_2 = torch.cat((out, residual_1), dim=1)  # 512
        concat_2 = torch.cat((concat_2,residual_2), dim = 1) # 512 + 256

        out = self.norm3(self.conv3(concat_2))
        out = self.nonlin3(out)




        return out




class DenseUpLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None, block=DenseUpBlock):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            block(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[block(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)




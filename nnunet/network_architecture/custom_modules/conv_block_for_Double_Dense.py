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








# IC conv 작업

# class ICConv3d(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size, padding, props, stride=1, bias=False):
#         super(ICConv3d, self).__init__()
#         self.conv_list = nn.ModuleList()
#         self.planes = planes
#         self.conv_1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,padding=padding, bias=bias, dilation=1)
#         self.conv_2 = nn.Conv3d(planes, planes, kernel_size=kernel_size,padding=2, bias=bias, dilation=2)
#         self.conv_3 = nn.Conv3d(planes, planes, kernel_size=kernel_size,padding=3, bias=bias,  dilation=3)
#
#         self.norm = props['norm_op'](planes, **props['norm_op_kwargs'])
#         self.nonlin = props['nonlin'](**props['nonlin_kwargs'])
#
#
#         self.conv_1_1 = nn.Conv3d(planes*3, planes, kernel_size=[1 for _ in kernel_size],
#                                       padding=[0 for i in kernel_size])
#         self.norm_1_1 = props['norm_op'](planes, **props['norm_op_kwargs'])
#         self.nonlin_1_1 = props['nonlin'](**props['nonlin_kwargs'])
#
#     def forward(self, x):
#
#
#         out_1 = self.nonlin(self.norm(self.conv_1(x)))
#         out_2 = self.nonlin(self.norm(self.conv_2(out_1)))
#         out_3 = self.nonlin(self.norm(self.conv_3(out_2)))
#
#         out = torch.cat((out_1, out_2, out_3), dim=1)
#
#         out = self.nonlin_1_1(self.norm_1_1(self.conv_1_1(out)))
#         return out





## D-Dense
class ICConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, props, stride=1, bias=False):
        super(ICConv3d, self).__init__()
        self.conv_list = nn.ModuleList()
        self.planes = planes
        self.conv_1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,padding=padding, bias=bias, dilation=1)
        self.conv_2 = nn.Conv3d((inplanes + planes), planes, kernel_size=kernel_size,padding=2, bias=bias, dilation=2)
        self.conv_3 = nn.Conv3d((inplanes + 2*planes), planes, kernel_size=kernel_size,padding=3, bias=bias,  dilation=3)

        self.norm = props['norm_op'](planes, **props['norm_op_kwargs'])
        self.nonlin = props['nonlin'](**props['nonlin_kwargs'])


        self.conv_1_1 = nn.Conv3d((inplanes + 3*planes), planes, kernel_size=[1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size])
        self.norm_1_1 = props['norm_op'](planes, **props['norm_op_kwargs'])
        self.nonlin_1_1 = props['nonlin'](**props['nonlin_kwargs'])

    def forward(self, x):


        out_1 = self.nonlin(self.norm(self.conv_1(x))) #32
        residual_1 = torch.cat((x,out_1), dim=1) # 32 + 32

        out_2 = self.nonlin(self.norm(self.conv_2(residual_1))) #32
        residual_2 = torch.cat((out_2,residual_1),dim=1) # 32 + 64

        out_3 = self.nonlin(self.norm(self.conv_3(residual_2))) # 32
        out = torch.cat((residual_2, out_3), dim=1) # 96 + 32

        out = self.nonlin_1_1(self.norm_1_1(self.conv_1_1(out))) # 32
        return out









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

        self.conv1 = ICConv3d(in_planes, in_planes, kernel_size=kernel_size,
                              padding=[(i - 1) // 2 for i in kernel_size], props=props)

        self.conv2 = ICConv3d(in_planes*2, in_planes, kernel_size=kernel_size,
                              padding=[(i - 1) // 2 for i in kernel_size], props=props)


    def forward(self, x):

        residual_1 = x  # 32
        out_1 = self.conv1(x)

        residual_2 = out_1
        concat_1 = torch.cat((out_1, residual_1), dim=1)  # 32 * 2

        residual_out = self.conv2(concat_1)  # 32 * 2
        out = self.pool_op(residual_out)

        return out , residual_out








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



        self.conv1 = ICConv3d(in_planes, in_planes, kernel_size=kernel_size,
                              padding=[(i - 1) // 2 for i in kernel_size], props=props)

        self.conv2 = ICConv3d(in_planes*2, in_planes*2, kernel_size=kernel_size,
                              padding=[(i - 1) // 2 for i in kernel_size], props=props)

        # conv3 = 1*1 conv
        self.conv3 = props['conv_op'](in_planes * 4, in_planes * 2, [1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size],
                                      **props['conv_op_kwargs'])

        self.norm3 = props['norm_op'](in_planes * 2, **props['norm_op_kwargs'])
        self.nonlin3 = props['nonlin'](**props['nonlin_kwargs'])


        # dropout
        props['dropout_op_kwargs'] = {'p': 0.5, 'inplace': True}
        props['dropout_op'] = nn.Dropout3d
        if props['dropout_op_kwargs']['p'] != 0:
            self.dropout = props['dropout_op'](**props['dropout_op_kwargs'])
        else:
            self.dropout = Identity()



    def forward(self, x):


        residual_1 = x  # 32

        # dropout
        # out_1 = self.nonlin1(self.norm1(self.dropout(self.conv1(x))))  # 32
        out_1 = self.conv1(x)  # 32
        residual_2 = out_1
        concat_1 = torch.cat((out_1, residual_1), dim=1)  # 32 * 2

        out = self.conv2(concat_1)  # 32 * 2



        concat_2 = torch.cat((out, residual_1), dim=1)  # 32*2 + 32*1 = 32 * 3
        concat_2 = torch.cat((concat_2,residual_2), dim = 1) # 32*3 + 32* = 32 * 4


        residual_out = self.nonlin3(self.norm3(self.conv3(concat_2)))
        out = self.pool_op(residual_out)




        return out ,residual_out





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

        self.conv1 = ICConv3d(aim_planes, aim_planes, kernel_size=kernel_size,
                              padding=[(i - 1) // 2 for i in kernel_size], props=props)


        self.conv2 = ICConv3d(aim_planes*2, aim_planes, kernel_size=kernel_size,
                              padding=[(i - 1) // 2 for i in kernel_size], props=props)


        self.conv3 = props['conv_op'](aim_planes * 3, aim_planes, [1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm3 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin3 = props['nonlin'](**props['nonlin_kwargs'])

        # dropout
        props['dropout_op_kwargs'] = {'p': 0.5, 'inplace': True}
        props['dropout_op'] = nn.Dropout3d
        if props['dropout_op_kwargs']['p'] != 0:
            self.dropout = props['dropout_op'](**props['dropout_op_kwargs'])
        else:
            self.dropout = Identity()








    def forward(self, x):


        x = self.nonlin0(self.norm0(self.conv0(x)))  # 256
        residual_1 = x  # 256

        # dropout
        # out_1 = self.nonlin1(self.norm1(self.dropout(self.conv1(x))))  # 256
        out_1 = self.conv1(x)  # 256
        residual_2 = out_1  # 256
        concat_1 = torch.cat((out_1, residual_1), dim=1)  # 512

        out = self.conv2(concat_1)  # 256


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














# normal Block

class PlainUpBlock(nn.Module):
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
            kwargs_conv1['stride'] = (1, 1, 1)
        else:
            kwargs_conv1 = props['conv_op_kwargs']


        # small version

        aim_planes = in_planes // 2  # 256

        self.conv0 = props['conv_op'](in_planes, aim_planes, [1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm0 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin0 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv1 = props['conv_op'](aim_planes, aim_planes, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv2 = props['conv_op'](aim_planes * 2, aim_planes, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])



    def forward(self, x):


        # small version
        x = self.nonlin0(self.norm0(self.conv0(x)))  # 256

        # x = self.nonlin1(self.norm1(self.conv1(x)))  # 256
        #
        # out = self.nonlin2(self.norm2(self.conv2(x)))  # 256

        return x




class PlainUpLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None, block=PlainUpBlock):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            block(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[block(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)









# Coordinate Attention

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()


        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_d = nn.AdaptiveAvgPool3d((1, 1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()

        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x   #(1,512,24,22,4)

        b, c, h, w, d = x.size()
        x_h = self.pool_h(x)  # (1,512,24,1,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2, 4) # (1,512,22,1,1)
        x_d = self.pool_d(x).permute(0, 1, 4, 3, 2) # (1,512,4,1,1)

        # print(x_h.shape)
        # print(x_w.shape)
        # print(x_d.shape)

        y = torch.cat([x_h, x_w, x_d], dim=2)   # (1,512,50,1,1)
        # print(y.shape)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) # (1,16,50,1,1)

        x_h, x_w, x_d = torch.split(y, [h, w, d], dim=2)
        x_w = x_w.permute(0, 1, 3, 2, 4)
        x_d = x_d.permute(0, 1, 4, 3, 2)


        a_d = self.conv_d(x_d).sigmoid()
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_d * a_w * a_h

        # 값이 줄어드는걸 보정하기 위해 3을 곱해준다
        # rst = out * 4
        # print(identity.max())
        # print(out.max())
        # print(rst.max())
        # print()

        return out # rst

    # tensor(15.6641, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
    # tensor(4.8398, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
    # tensor(14.5156, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)

    # tensor(19.5781, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
    # tensor(4.7500, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
    # tensor(14.2500, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)

    # tensor(6.5195, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
    # tensor(1.2051, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
    # tensor(3.6152, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)

    # tensor(8.0078, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
    # tensor(2.1250, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
    # tensor(6.3750, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)

    # tensor(10.0859, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
    # tensor(2.0781, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
    # tensor(6.2344, device='cuda:0', dtype=torch.float16, grad_fn= < MaxBackward1 >)
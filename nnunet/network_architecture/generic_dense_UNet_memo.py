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


import numpy as np
import torch
from nnunet.network_architecture.custom_modules.conv_blocks import DenseDownLayer, DenseDownBlock, DenseUpBlock, DenseUpLayer
# from nnunet.network_architecture.custom_modules.conv_blocks import BasicDenseBlock
from nnunet.network_architecture.generic_UNet import Upsample
from nnunet.network_architecture.generic_modular_UNet import PlainConvUNetDecoder, get_default_network_config
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn
from torch.optim import SGD
from torch.backends import cudnn


class DenseUNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, default_return_skips=True,
                 max_num_features=480, block=DenseDownBlock):
        """
        Following UNet building blocks can be added by utilizing the properties this class exposes (TODO)

        this one includes the bottleneck layer!

        :param input_channels:
        :param base_num_features:
        :param num_blocks_per_stage:
        :param feat_map_mul_on_downscale:
        :param pool_op_kernel_sizes:
        :param conv_kernel_sizes:
        :param props:
        """
        super(DenseUNetEncoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.props = props

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        num_stages = len(conv_kernel_sizes)

        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages

        self.num_blocks_per_stage = num_blocks_per_stage  # decoder may need this

        self.initial_conv = props['conv_op'](input_channels, base_num_features, 3, padding=1, **props['conv_op_kwargs'])
        self.initial_norm = props['norm_op'](base_num_features, **props['norm_op_kwargs'])
        self.initial_nonlin = props['nonlin'](**props['nonlin_kwargs'])

        current_input_features = base_num_features
        for stage in range(1, num_stages+1):
            current_output_features = min(base_num_features * feat_map_mul_on_downscale ** stage, max_num_features)
            current_kernel_size = conv_kernel_sizes[stage-1]
            current_pool_kernel_size = pool_op_kernel_sizes[stage-1]

            current_stage = DenseDownLayer(current_input_features, current_output_features, current_kernel_size, props,
                                          self.num_blocks_per_stage[stage-1], current_pool_kernel_size, block)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)

    def forward(self, x, return_skips=None):
        """

        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []

        x = self.initial_nonlin(self.initial_norm(self.initial_conv(x)))
        for s in self.stages:
            x = s(x)
            if self.default_return_skips:
                skips.append(x)

        if return_skips is None:
            return_skips = self.default_return_skips

        if return_skips:
            return skips
        else:
            return x

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, pool_op_kernel_sizes, num_conv_per_stage_encoder,
                                        feat_map_mul_on_downscale, batch_size):
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)

        tmp = (num_conv_per_stage_encoder[0] * 2 + 1) * np.prod(current_shape) * base_num_features \
              + num_modalities * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool + 1):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_conv_per_stage_encoder[p] * 2 + 1  # + 1 for conv in skip in first block
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat
        return tmp * batch_size


class DenseUNetDecoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_stage=None, network_props=None, deep_supervision=False,
                 upscale_logits=False, block=DenseUpBlock):
        super(DenseUNetDecoder, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        """
        We assume the bottleneck is part of the encoder, so we can start with upsample -> concat here
        """
        previous_stages = previous.stages
        # previous_stage_output_features = previous.stage_output_features
        previous_stage_output_features = [32,64,128,256,512]
        previous_stage_pool_kernel_size = [[1,1,1], [1,1,1], [1,1,1], [1,1,1],[1,1,1]]

        # previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size

        if network_props is None:
            self.props = previous.props
        else:
            self.props = network_props

        if self.props['conv_op'] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props['conv_op'] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.props['conv_op']))

        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]

        # assert len(num_blocks_per_stage) == len(previous.num_blocks_per_stage) - 1

        num_blocks_per_stage = (1,1,1,1,1)



        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size

        # num_stages = len(previous_stages) - 1  # we have one less as the first stage here is what comes after the
        num_stages = len(previous_stages)
        # bottleneck

        self.tus = []
        self.stages = []
        self.deep_supervision_outputs = []

        # only used for upsample_logits
        cum_upsample = np.cumprod(np.vstack(self.stage_pool_kernel_size), axis=0).astype(int)

        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s+1]
            features_skip = previous_stage_output_features[s]

            self.tus.append(transpconv(features_below, features_skip, previous_stage_pool_kernel_size[s + 1],
                                       previous_stage_pool_kernel_size[s+1], bias=False))
            # after we tu we concat features so now we have 2xfeatures_skip
            self.stages.append(DenseUpLayer(2 * features_skip, features_skip, previous_stage_conv_op_kernel_size[s],
                                             self.props, num_blocks_per_stage[i], None, block))

            if deep_supervision and s != 0:
                seg_layer = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, False)
                if upscale_logits:
                    upsample = Upsample(scale_factor=cum_upsample[s], mode=upsample_mode)
                    self.deep_supervision_outputs.append(nn.Sequential(seg_layer, upsample))
                else:
                    self.deep_supervision_outputs.append(seg_layer)

        self.segmentation_output = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, False)

        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, skips):
        # skips come from the encoder. They are sorted so that the bottleneck is last in the list
        # what is maybe not perfect is that the TUs and stages here are sorted the other way around
        # so let's just reverse the order of skips
        skips = skips[::-1]
        seg_outputs = []

        x = skips[0]  # this is the bottleneck

        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)
            if self.deep_supervision and (i != len(self.tus) - 1):
                seg_outputs.append(self.deep_supervision_outputs[i](x))

        segmentation = self.segmentation_output(x)

        if self.deep_supervision:
            seg_outputs.append(segmentation)
            return seg_outputs[
                   ::-1]  # seg_outputs are ordered so that the seg from the highest layer is first, the seg from
            # the bottleneck of the UNet last
        else:
            return segmentation

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_classes, pool_op_kernel_sizes, num_blocks_per_stage_decoder,
                                        feat_map_mul_on_downscale, batch_size):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :return:
        """
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)
        tmp = (num_blocks_per_stage_decoder[-1] * 2 + 1) * np.prod(
            current_shape) * base_num_features + num_classes * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_blocks_per_stage_decoder[-(p + 1)] * 2 + 1 + 1  # +1 for transpconv and +1 for conv in skip
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat

        return tmp * batch_size


class DenseUNet(SegmentationNetwork):
    """
    Residual Encoder, Plain conv decoder
    """
    use_this_for_2D_configuration = 1244233721.0  # 1167982592.0
    use_this_for_3D_configuration = 1230348801.0
    # default_blocks_per_stage_encoder = (1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4)
    default_blocks_per_stage_encoder = (1, 1, 1, 1, 1, 1, 1, 1)
    default_blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1, 1, 1)
    default_min_batch_size = 4  # this is what works with the numbers above

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None,
                 block=DenseDownBlock,
                 props_decoder=None):
        super().__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes


        props['dropout_op_kwargs'] = {'p': 0.5, 'inplace': True}
        props['dropout_op'] = nn.Dropout3d


        self.encoder = DenseUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                           feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                           props, default_return_skips=True, max_num_features=max_features,
                                           block=block)
       #  props['dropout_op_kwargs']['p'] = 0



        if props_decoder is None:
            props_decoder = props
        self.decoder = DenseUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props_decoder,
                                            deep_supervision, upscale_logits, block=DenseUpBlock)
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes,
                                        num_conv_per_stage_encoder,
                                        num_conv_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
        enc = DenseUNetEncoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                  num_modalities, pool_op_kernel_sizes,
                                                                  num_conv_per_stage_encoder,
                                                                  feat_map_mul_on_downscale, batch_size)
        dec = DenseUNetDecoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                   num_classes, pool_op_kernel_sizes,
                                                                   num_conv_per_stage_decoder,
                                                                   feat_map_mul_on_downscale, batch_size)

        return enc + dec









if __name__ == "__main__":
    pass


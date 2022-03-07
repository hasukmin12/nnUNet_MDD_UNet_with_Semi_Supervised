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
from nnunet.network_architecture.custom_modules.conv_block_for_Double_Dense import DenseDownLayer_2, DenseDownBlock_2, DenseUpBlock, DenseUpLayer , DenseDownLayer_first, DenseDownBlock_first
from nnunet.network_architecture.generic_UNet import Upsample
from nnunet.network_architecture.generic_modular_UNet import PlainConvUNetDecoder, get_default_network_config
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn
from torch.optim import SGD
from torch.backends import cudnn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets import ViT
from typing import Tuple, Union


class UNETR_Encoder_Decoder(nn.Module):
    def __init__(self, in_channels, output_channels, props,
                 img_size: Tuple[int, int, int],
                 feature_size = 16, hidden_size = 768, mlp_dim = 3072, num_heads = 12,
                 pos_embed = 'perceptron', norm_name:Union[Tuple, str] = "instance",
                 conv_block:bool = False,
                 res_block:bool = True,
                 dropout_rate:float = 0.0,
                 default_return_skips=True,
                 deep_supervision=False):


        """
        :input_channels: (1)
        :output_channels: (2) = 0,1의 binary segmentation으로 최종 아웃풋의 클래스 수를 뜻함
        :img_size : (96,96,32) 패치 사이즈
        :feature_size : dimension of network feature size.
        :hidden_size: dimension of hidden layer.
        :mlp_dim: dimension of feedforward layer.함
        :num_heads: number of attention heads.
        :pos_embed: position embedding layer type.
        :norm_name: feature normalization type and arguments.
        :conv_block: bool argument to determine if convolutional block is used.
        :res_block: bool argument to determine if residual block is used.
        :dropout_rate: faction of the input units to drop.

        :num_blocks_per_stage: (1,1,1,1) -> ViT에서는 불필요
        :feat_map_mul_on_downscale:  (2) -> ? feature map의 채널수가 곱해지는 비율 (2로 설정됬으니 32 -> 64로감) -> ViT에서는 불필요
        :pool_op_kernel_sizes: [[2,2,2],[2,2,2],[2,2,2],[2,2,2]] -> ViT에서는 불필요
        :conv_kernel_sizes: [[3,3,3],[3,3,3],[3,3,3],[3,3,3]] -> ViT에서는 불필요
        :props:
        """


        super(UNETR_Encoder_Decoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.props = props
        self.deep_supervision = deep_supervision

        self.stages = []
        # self.stage_output_features = []
        # self.stage_pool_kernel_size = []
        # self.stage_conv_op_kernel_size = []

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.stages.append(self.vit)
        self.stages.append(self.encoder1)
        self.stages.append(self.encoder2)
        self.stages.append(self.encoder3)
        self.stages.append(self.encoder4)
        self.stages = nn.ModuleList(self.stages)


        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out_1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=output_channels)
        self.out_2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size*2, out_channels=output_channels)
        self.out_3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size*4, out_channels=output_channels)

        self.stages.append(self.decoder5)
        self.stages.append(self.decoder4)
        self.stages.append(self.decoder3)
        self.stages.append(self.decoder2)
        self.stages = nn.ModuleList(self.stages)

        self.deep_supervision_outputs = []
        self.deep_supervision_outputs.append(self.out_1)
        self.deep_supervision_outputs.append(self.out_2)
        self.deep_supervision_outputs.append(self.out_3)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)



    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights['state_dict']:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights['state_dict']['module.transformer.patch_embedding.position_embeddings_3d'])
            self.vit.patch_embedding.cls_token.copy_(
                weights['state_dict']['module.transformer.patch_embedding.cls_token'])
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.weight'])
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.bias'])

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights['state_dict']['module.transformer.norm.weight'])
            self.vit.norm.bias.copy_(weights['state_dict']['module.transformer.norm.bias'])


    # x : (1,1,192,176,32)
    # RuntimeError: The size of tensor a (264) must match the size of tensor b (72) at non-singleton dimension 1

    def forward(self, x_in):
        out_list = []
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        out_3 = self.out_3(dec2)

        dec1 = self.decoder3(dec2, enc2)
        out_2 = self.out_2(dec1)

        out = self.decoder2(dec1, enc1)
        out_1 = self.out_1(out)

        out_list.append(out_1)
        out_list.append(out_2)
        out_list.append(out_3)

        return out_list



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






class UNETR(SegmentationNetwork):
    """
    Residual Encoder, Plain conv decoder
    """
    use_this_for_2D_configuration = 1244233721.0  # 1167982592.0
    use_this_for_3D_configuration = 1230348801.0
    default_blocks_per_stage_encoder = (1, 1, 1, 1, 1, 1, 1, 1)
    default_blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1, 1, 1)
    default_min_batch_size = 4  # this is what works with the numbers above

    def __init__(self, in_channels, num_classes, props,
                 img_size: Tuple[int, int, int],
                 feature_size = 16, hidden_size = 768, mlp_dim = 3072, num_heads = 12,
                 pos_embed = 'perceptron', norm_name:Union[Tuple, str] = "instance",
                 conv_block:bool = False,
                 res_block:bool = True,
                 dropout_rate:float = 0.0,
                 default_return_skips=True):

                 # input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 # pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 # deep_supervision=False, upscale_logits=False, max_features=512, initializer=None,
                 # block=DenseDownBlock_2,
                 # props_decoder=None):


        super().__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes
        self.encoder_decoder = UNETR_Encoder_Decoder(in_channels,num_classes, props, img_size, feature_size, hidden_size, mlp_dim,
                                                     num_heads, pos_embed, norm_name, conv_block, res_block, dropout_rate, default_return_skips)



        # if initializer is not None:
        #     self.apply(initializer)

    def forward(self, x):
        return self.encoder_decoder(x)





    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes,
                                        num_conv_per_stage_encoder,
                                        num_conv_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
        rst = UNETR_Encoder_Decoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                  num_modalities, pool_op_kernel_sizes,
                                                                  num_conv_per_stage_encoder,
                                                                  feat_map_mul_on_downscale, batch_size)

        return rst





















def find_3d_configuration():
    # lets compute a reference for 3D
    # we select hyperparameters here so that we get approximately the same patch size as we would get with the
    # regular unet. This is just my choice. You can do whatever you want
    # These default hyperparemeters will then be used by the experiment planner

    # since this is more parameter intensive than the UNet, we will test a configuration that has a lot of parameters
    # herefore we copy the UNet configuration for Task005_Prostate
    cudnn.deterministic = False
    cudnn.benchmark = True

    patch_size = (20, 320, 256)
    max_num_features = 320
    num_modalities = 2
    num_classes = 3
    batch_size = 2

    # now we fiddle with the network specific hyperparameters until everything just barely fits into a titanx
    blocks_per_stage_encoder = UNETR.default_blocks_per_stage_encoder
    blocks_per_stage_decoder = UNETR.default_blocks_per_stage_decoder
    initial_num_features = 32

    # we neeed to add a [1, 1, 1] for the res unet because in this implementation all stages of the encoder can have a stride
    pool_op_kernel_sizes = [[1, 1, 1],
                            [1, 2, 2],
                            [1, 2, 2],
                            [2, 2, 2],
                            [2, 2, 2],
                            [1, 2, 2],
                            [1, 2, 2]]

    conv_op_kernel_sizes = [[1, 3, 3],
                            [1, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3]]

    unet = UNETR(num_modalities, initial_num_features, blocks_per_stage_encoder[:len(conv_op_kernel_sizes)], 2,
                       pool_op_kernel_sizes, conv_op_kernel_sizes,
                       get_default_network_config(3, dropout_p=None), num_classes,
                       blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], False, False,
                       max_features=max_num_features).cuda()

    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)
    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    dummy_input = torch.rand((batch_size, num_modalities, *patch_size)).cuda()
    dummy_gt = (torch.rand((batch_size, 1, *patch_size)) * num_classes).round().clamp_(0, 2).cuda().long()

    for _ in range(20):
        optimizer.zero_grad()
        skips = unet.encoder(dummy_input)
        print([i.shape for i in skips])
        output = unet.decoder(skips)

        l = loss(output, dummy_gt)
        l.backward()

        optimizer.step()
        if _ == 0:
            torch.cuda.empty_cache()

    # that should do. Now take the network hyperparameters and insert them in FabiansUNet.compute_approx_vram_consumption
    # whatever number this spits out, save it to FabiansUNet.use_this_for_batch_size_computation_3D
    print(UNETR.compute_approx_vram_consumption(patch_size, initial_num_features, max_num_features, num_modalities,
                                                num_classes, pool_op_kernel_sizes,
                                                blocks_per_stage_encoder[:len(conv_op_kernel_sizes)],
                                                blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], 2, batch_size))
    # the output is 1230348800.0 for me
    # I increment that number by 1 to allow this configuration be be chosen


if __name__ == "__main__":
    pass


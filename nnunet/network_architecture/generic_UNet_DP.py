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
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import get_tp_fp_fn_tn
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn


class Generic_UNet_DP(Generic_UNet):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None):
        """
        As opposed to the Generic_UNet, this class will compute parts of the loss function in the forward pass. This is
        useful for GPU parallelization. The batch DICE loss, if used, must be computed over the whole batch. Therefore, in a
        naive implementation, all softmax outputs must be copied to a single GPU which will then
        do the loss computation all by itself. In the context of 3D Segmentation, this results in a lot of overhead AND
        is inefficient because the DICE computation is also kinda expensive (Think 8 GPUs with a result of shape
        2x4x128x128x128 each.). The DICE is a global metric, but its parts can be computed locally (TP, FP, FN). Thus,
        this implementation will compute all the parts of the loss function in the forward pass (and thus in a
        parallelized way). The results are very small (batch_size x num_classes for TP, FN and FP, respectively; scalar for CE) and
        copied easily. Also the final steps of the loss function (computing batch dice and average CE values) are easy
        and very quick on the one GPU they need to run on. BAM.
        final_nonlin is lambda x:x here!
        """
        super(Generic_UNet_DP, self).__init__(input_channels, base_num_features, num_classes, num_pool,
                                              num_conv_per_stage,
                                              feat_map_mul_on_downscale, conv_op,
                                              norm_op, norm_op_kwargs,
                                              dropout_op, dropout_op_kwargs,
                                              nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
                                              lambda x: x, weightInitializer, pool_op_kernel_sizes,
                                              conv_kernel_sizes,
                                              upscale_logits, convolutional_pooling, convolutional_upsampling,
                                              max_num_features)
        self.ce_loss = RobustCrossEntropyLoss()


# 여기서 x는 input(CT 영상), y는 label된 GT(Label GT)이다.

# 각각의 res[i], y[i] 값은 sampling하는 Layer 각각의 output을 뜻한다.
# 각각의 i 값이 배치라고 생각하면 된다.
# 여기서 각각의 res 및 y의 텐서 크기는 (40*40)이다.
# 본래의 영상 및 라벨의 사이즈를 줄이고 분할해서 (40*40)로 만든거라고 이해하면 된다.

    def forward(self, x, y=None, return_hard_tp_fp_fn=False):
        res = super(Generic_UNet_DP, self).forward(x)  # regular Generic_UNet forward pass


        # print("res[0].shape: ", res[0].shape)
        # print("res[1].shape : ", res[1].shape)
        # print("res[2].shape : ", res[2].shape)
        # print("res[3].shape : ", res[3].shape)
        #
        # print("res[0].max()", res[0].max())
        # print("res[1].max()", res[1].max())
        #
        #
        #
        # print("y[0].shape: ",y[0].shape)
        # print("y[1].shape : ",y[1].shape)
        # print("y[2].shape : ", y[2].shape)
        # print("y[3].shape : ", y[3].shape)
        #
        # print("y[0].max()", y[0].max())
        # print("y[1].max()", y[1].max())



        if y is None:
            return res

        # 여기로 들어간다고 보면 된다.
        else:
            # compute ce loss
            # print("compute ce loss")

            # self._deep_supervision = True
            if self._deep_supervision and self.do_ds:
                # print("deep_supervision is True")
                ce_losses = [self.ce_loss(res[0], y[0]).unsqueeze(0)]
                # ce_losses_0 = [self.ce_loss(res[0][:][0], y[0]).unsqueeze(0)]
                # ce_losses_1 = [self.ce_loss(res[0][:][1], y[0]).unsqueeze(0)]

                # print("ce_losses : ", ce_losses.shape)




                # tp : True Positive
                # fp : False Positive
                # fn : False Negative
                tps = []
                fps = []
                fns = []

                res_softmax = softmax_helper(res[0])
               # print("res_softmax.shape : ", res_softmax.shape)
                tp, fp, fn, _ = get_tp_fp_fn_tn(res_softmax, y[0])
                tps.append(tp)
                fps.append(fp)
                fns.append(fn)
                for i in range(1, len(y)):
                    ce_losses.append(self.ce_loss(res[i], y[i]).unsqueeze(0))
                    res_softmax = softmax_helper(res[i])
                    tp, fp, fn, _ = get_tp_fp_fn_tn(res_softmax, y[i])
                    tps.append(tp)
                    fps.append(fp)
                    fns.append(fn)
                ret = ce_losses, tps, fps, fns

            # 여긴 안쓰인다고 보면된다.
            else:
                # print("deep_supervision is False")
                ce_loss = self.ce_loss(res, y).unsqueeze(0)

                # tp fp and fn need the output to be softmax
                res_softmax = softmax_helper(res)

                tp, fp, fn, _ = get_tp_fp_fn_tn(res_softmax, y)

                ret = ce_loss, tp, fp, fn


            # validation 시 여기로 들어간다.
            if return_hard_tp_fp_fn:

                # print("return_hard_tp_fp_fn is True")
                if self._deep_supervision and self.do_ds:
                    output = res[0]
                    target = y[0]
                else:
                    target = y
                    output = res

                with torch.no_grad():
                    num_classes = output.shape[1]
                    output_softmax = softmax_helper(output)
                    output_seg = output_softmax.argmax(1)
                    target = target[:, 0]
                    axes = tuple(range(1, len(target.shape)))
                    tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                    fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                    fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                    for c in range(1, num_classes):
                        tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                        fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                        fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

                    tp_hard = tp_hard.sum(0, keepdim=False)[None]
                    fp_hard = fp_hard.sum(0, keepdim=False)[None]
                    fn_hard = fn_hard.sum(0, keepdim=False)[None]

                    ret = *ret, tp_hard, fp_hard, fn_hard
            return ret

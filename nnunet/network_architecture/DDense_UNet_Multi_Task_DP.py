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
from nnunet.network_architecture.DDense_UNet_Multi_Task import D_DenseUNet_Multi_Task
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import get_tp_fp_fn_tn
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.network_architecture.custom_modules.conv_block_for_Double_Dense import DenseUpBlock, DenseUpLayer, DenseDownBlock_2, DenseDownLayer_2

from torch import nn


class D_DenseUNet_Multi_Task_DP(D_DenseUNet_Multi_Task):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None,
                 block=DenseDownBlock_2):


        super(D_DenseUNet_Multi_Task_DP, self).__init__(input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None,
                 block=DenseDownBlock_2)   # , lambda x: x)



        self.ce_loss = RobustCrossEntropyLoss()


# 여기서 x는 input(CT 영상), y는 label된 GT(Label GT)이다.

# 각각의 res[i], y[i] 값은 sampling하는 Layer 각각의 output을 뜻한다.
# 각각의 i 값이 배치라고 생각하면 된다.
# 여기서 각각의 res 및 y의 텐서 크기는 (40*40)이다.
# 본래의 영상 및 라벨의 사이즈를 줄이고 분할해서 (40*40)로 만든거라고 이해하면 된다.


    def forward(self, x, y1=None, y2=None, y3=None, return_hard_tp_fp_fn=False):
        out1, out2, out3 = super(D_DenseUNet_Multi_Task_DP, self).forward(x)
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

        if y1 is None and y2 is None and y3 is None:
            return out1, out2, out3

            # 여기로 들어간다고 보면 된다.
        else:
            # compute ce loss
            # print("compute ce loss")

            self._deep_supervision = True
            if self._deep_supervision and self.decoder1.deep_supervision:
                ce_losses1 = [self.ce_loss((out1[0]), y1[0]).unsqueeze(0)]
                # tp : True Positive
                # fp : False Positive
                # fn : False Negative
                tps1 = []
                fps1 = []
                fns1 = []
                res_softmax1 = softmax_helper(out1[0])
                tp1, fp1, fn1, _ = get_tp_fp_fn_tn(res_softmax1, y1[0])
                tps1.append(tp1)
                fps1.append(fp1)
                fns1.append(fn1)
                for i in range(1, len(y1)):
                    ce_losses1.append(self.ce_loss(out1[i], y1[i]).unsqueeze(0))
                    res_softmax1 = softmax_helper(out1[i])
                    tp1, fp1, fn1, _ = get_tp_fp_fn_tn(res_softmax1, y1[i])
                    tps1.append(tp1)
                    fps1.append(fp1)
                    fns1.append(fn1)
                ret1 = ce_losses1, tps1, fps1, fns1

                ce_losses2 = [self.ce_loss(out2[0], y2[0]).unsqueeze(0)]
                # tp : True Positive
                # fp : False Positive
                # fn : False Negative
                tps2 = []
                fps2 = []
                fns2 = []
                res_softmax2 = softmax_helper(out2[0])
                tp2, fp2, fn2, _ = get_tp_fp_fn_tn(res_softmax2, y2[0])
                tps2.append(tp2)
                fps2.append(fp2)
                fns2.append(fn2)
                for i in range(1, len(y2)):
                    ce_losses2.append(self.ce_loss(out2[i], y2[i]).unsqueeze(0))
                    res_softmax2 = softmax_helper(out2[i])
                    tp2, fp2, fn2, _ = get_tp_fp_fn_tn(res_softmax2, y2[i])
                    tps2.append(tp2)
                    fps2.append(fp2)
                    fns2.append(fn2)
                ret2 = ce_losses2, tps2, fps2, fns2

                ce_losses3 = [self.ce_loss(out3[0], y3[0]).unsqueeze(0)]
                # tp : True Positive
                # fp : False Positive
                # fn : False Negative
                tps3 = []
                fps3 = []
                fns3 = []
                res_softmax3 = softmax_helper(out3[0])
                tp3, fp3, fn3, _ = get_tp_fp_fn_tn(res_softmax3, y3[0])
                tps3.append(tp3)
                fps3.append(fp3)
                fns3.append(fn3)
                for i in range(1, len(y3)):
                    ce_losses3.append(self.ce_loss(out3[i], y3[i]).unsqueeze(0))
                    res_softmax3 = softmax_helper(out3[i])
                    tp3, fp3, fn3, _ = get_tp_fp_fn_tn(res_softmax3, y3[i])
                    tps3.append(tp3)
                    fps3.append(fp3)
                    fns3.append(fn3)
                ret3 = ce_losses3, tps3, fps3, fns3

            else:
                print("deep_supervision is False")
                ce_loss1 = self.ce_loss(out1, y1).unsqueeze(0)
                # tp fp and fn need the output to be softmax
                res_softmax1 = softmax_helper(out1)
                tp1, fp1, fn1, _ = get_tp_fp_fn_tn(res_softmax1, y1)
                ret1 = ce_loss1, tp1, fp1, fn1

                ce_loss2 = self.ce_loss(out2, y2).unsqueeze(0)
                # tp fp and fn need the output to be softmax
                res_softmax2 = softmax_helper(out2)
                tp2, fp2, fn2, _ = get_tp_fp_fn_tn(res_softmax2, y2)
                ret2 = ce_loss2, tp2, fp2, fn2

                ce_loss3 = self.ce_loss(out3, y3).unsqueeze(0)
                # tp fp and fn need the output to be softmax
                res_softmax3 = softmax_helper(out3)
                tp3, fp3, fn3, _ = get_tp_fp_fn_tn(res_softmax3, y3)
                ret3 = ce_loss3, tp3, fp3, fn3

                # validation 시 여기로 들어간다.
            if return_hard_tp_fp_fn:

                # print("return_hard_tp_fp_fn is True")
                if self._deep_supervision and self.decoder1.deep_supervision:
                    output1 = out1[0]
                    target1 = y1[0]
                    # print("target1.max() : ",target1.max())

                    output2 = out2[0]
                    target2 = y2[0]
                    # print("target2.max() : ", target2.max())

                    output3 = out3[0]
                    target3 = y3[0]
                    # print("target3.max() : ", target3.max())

                else:
                    target1 = y1
                    output1 = out1
                    target2 = y2
                    output2 = out2
                    target3 = y3
                    output3 = out3

                with torch.no_grad():
                    num_classes = output1.shape[1]
                    output_softmax = softmax_helper(output1)
                    output_seg = output_softmax.argmax(1)
                    target = target1[:, 0]
                    # print("target.shape : ", target.shape)

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

                    ret1 = *ret1, tp_hard, fp_hard, fn_hard

                    del num_classes, output_softmax, output_seg, target, axes, tp_hard, fp_hard, fn_hard

                    num_classes = output2.shape[1]
                    output_softmax = softmax_helper(output2)
                    output_seg = output_softmax.argmax(1)
                    target = target2[:, 0]
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

                    ret2 = *ret2, tp_hard, fp_hard, fn_hard

                    del num_classes, output_softmax, output_seg, target, axes, tp_hard, fp_hard, fn_hard

                    num_classes = output3.shape[1]
                    output_softmax = softmax_helper(output3)
                    output_seg = output_softmax.argmax(1)
                    target = target3[:, 0]
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

                    ret3 = *ret3, tp_hard, fp_hard, fn_hard

                    del num_classes, output_softmax, output_seg, target, axes, tp_hard, fp_hard, fn_hard

            return ret1, ret2, ret3


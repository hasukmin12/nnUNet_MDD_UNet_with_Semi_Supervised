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
from nnunet.network_architecture.generic_UNet_Multi_Task import Generic_UNet_Multi_Task
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import get_tp_fp_fn_tn
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda, to_cuda_1, to_cuda_2, to_cuda_3


class Generic_UNet_Multi_Task_DP(Generic_UNet_Multi_Task):
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
        super(Generic_UNet_Multi_Task_DP, self).__init__(input_channels, base_num_features, num_classes, num_pool,
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

    def forward(self, x, y1=None, y2=None, y3=None, return_hard_tp_fp_fn=False):
        out1, out2, out3 = super(Generic_UNet_Multi_Task_DP, self).forward(x)  # regular Generic_UNet forward pass

        # y1 = y
        # y2 = y
        # y3 = y
        #
        # y1 = to_cuda_1(y1)
        # y2 = to_cuda_2(y2)
        # y3 = to_cuda_3(y3)
        #
        # # y1, y2, y3를 각각 1,2,3의 요소만 남기고 모두 0으로 만든다.
        # for i in range(4):
        #     y1[i] = torch.where(y1[i] == 1, y1[i], to_cuda_1(torch.tensor(0).float()))
        #     y2[i] = torch.where(y2[i] == 2, y2[i], to_cuda_2(torch.tensor(0).float()))
        #     y3[i] = torch.where(y3[i] == 3, y3[i], to_cuda_3(torch.tensor(0).float()))
        #     # y1[i] = torch.where(y1[i] == 1, y1[i], torch.tensor(0).float())
        #     # y2[i] = torch.where(y2[i] == 2, y2[i], torch.tensor(0).float())
        #     # y3[i] = torch.where(y3[i] == 3, y3[i], torch.tensor(0).float())


        if y1 is None and y2 is None and y3 is None:
            return out1, out2, out3

        # 여기로 들어간다고 보면 된다.
        else:
            # compute ce loss
            # print("compute ce loss")

            self._deep_supervision = True
            if self._deep_supervision and self.do_ds:
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

            # if self._deep_supervision and self.do_ds:
            #     ce_losses1 = [self.ce_loss((to_cuda_1(out1[0])), y1[0]).unsqueeze(0)]
            #     # tp : True Positive
            #     # fp : False Positive
            #     # fn : False Negative
            #     tps1 = []
            #     fps1 = []
            #     fns1 = []
            #     res_softmax1 = softmax_helper(to_cuda_1(out1[0]))
            #     tp1, fp1, fn1, _ = get_tp_fp_fn_tn(res_softmax1, y1[0])
            #     tps1.append(tp1)
            #     fps1.append(fp1)
            #     fns1.append(fn1)
            #     for i in range(1, len(y1)):
            #         ce_losses1.append(self.ce_loss(to_cuda_1(out1[i]), y1[i]).unsqueeze(0))
            #         res_softmax1 = softmax_helper(to_cuda_1(out1[i]))
            #         tp1, fp1, fn1, _ = get_tp_fp_fn_tn(res_softmax1, y1[i])
            #         tps1.append(tp1)
            #         fps1.append(fp1)
            #         fns1.append(fn1)
            #     ret1 = ce_losses1, tps1, fps1, fns1
            #
            #     ce_losses2 = [self.ce_loss(to_cuda_2(out2[0]), y2[0]).unsqueeze(0)]
            #     # tp : True Positive
            #     # fp : False Positive
            #     # fn : False Negative
            #     tps2 = []
            #     fps2 = []
            #     fns2 = []
            #     res_softmax2 = softmax_helper(to_cuda_2(out2[0]))
            #     tp2, fp2, fn2, _ = get_tp_fp_fn_tn(res_softmax2, y2[0])
            #     tps2.append(tp2)
            #     fps2.append(fp2)
            #     fns2.append(fn2)
            #     for i in range(1, len(y2)):
            #         ce_losses2.append(self.ce_loss(out2[i], y2[i]).unsqueeze(0))
            #         res_softmax2 = softmax_helper(out2[i])
            #         tp2, fp2, fn2, _ = get_tp_fp_fn_tn(res_softmax2, y2[i])
            #         tps2.append(tp2)
            #         fps2.append(fp2)
            #         fns2.append(fn2)
            #     ret2 = ce_losses2, tps2, fps2, fns2
            #
            #     ce_losses3 = [self.ce_loss(out3[0], y3[0]).unsqueeze(0)]
            #     # tp : True Positive
            #     # fp : False Positive
            #     # fn : False Negative
            #     tps3 = []
            #     fps3 = []
            #     fns3 = []
            #     res_softmax3 = softmax_helper(out3[0])
            #     tp3, fp3, fn3, _ = get_tp_fp_fn_tn(res_softmax3, y3[0])
            #     tps3.append(tp3)
            #     fps3.append(fp3)
            #     fns3.append(fn3)
            #     for i in range(1, len(y3)):
            #         ce_losses3.append(self.ce_loss(out3[i], y3[i]).unsqueeze(0))
            #         res_softmax3 = softmax_helper(out3[i])
            #         tp3, fp3, fn3, _ = get_tp_fp_fn_tn(res_softmax3, y3[i])
            #         tps3.append(tp3)
            #         fps3.append(fp3)
            #         fns3.append(fn3)
            #     ret3 = ce_losses3, tps3, fps3, fns3



            # 여긴 언제 쓰이누
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
                if self._deep_supervision and self.do_ds:
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

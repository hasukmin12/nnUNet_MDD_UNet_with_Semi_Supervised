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
import copy

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.generic_UNet_Multi_Task_DP_in_one import Generic_UNet_Multi_Task_DP_in_one
from nnunet.network_architecture.DDense_UNet_Multi_Task_DP  import D_DenseUNet_Multi_Task_DP
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.network_training.nnUNetTrainerV2_3_branch import nnUNetTrainerV2_3_branch
# from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

from nnunet.utilities.to_torch import maybe_to_torch, to_cuda, to_cuda_1, to_cuda_2, to_cuda_3
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
# from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.cuda.amp import autocast
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.utils import clip_grad_norm_
from nnunet.training.network_training.nnUNetTrainer_3_branch import nnUNetTrainer_3_branch

from nnunet.network_architecture.DDense_UNet_Multi_Task import get_default_network_config

# 3 brnach with 3D-UNet

# nnUNetTrainerV2_3_branch

class HasTrainer_DP_DDense_3_branch(nnUNetTrainerV2_3_branch):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, num_gpus=1, distribute_batch_size=False, fp16=False):
        super(HasTrainer_DP_DDense_3_branch, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                                unpack_data, deterministic, fp16)
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, num_gpus, distribute_batch_size, fp16)
        self.num_gpus = num_gpus
        self.distribute_batch_size = distribute_batch_size
        self.dice_smooth = 1e-5
        self.dice_do_BG = False
        self.loss = None
        self.loss_weights = None



    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

    def run_online_evaluation(self, tp_hard, fp_hard, fn_hard):
        tp_hard = tp_hard.detach().cpu().numpy().mean(0)
        fp_hard = fp_hard.detach().cpu().numpy().mean(0)
        fn_hard = fn_hard.detach().cpu().numpy().mean(0)
        self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
        self.online_eval_tp.append(list(tp_hard))
        self.online_eval_fp.append(list(fp_hard))
        self.online_eval_fn.append(list(fn_hard))

    def run_online_evaluation_2(self, tp_hard, fp_hard, fn_hard):
        tp_hard = tp_hard.detach().cpu().numpy().mean(0)
        fp_hard = fp_hard.detach().cpu().numpy().mean(0)
        fn_hard = fn_hard.detach().cpu().numpy().mean(0)
        self.online_eval_foreground_dc_2.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
        self.online_eval_tp_2.append(list(tp_hard))
        self.online_eval_fp_2.append(list(fp_hard))
        self.online_eval_fn_2.append(list(fn_hard))

    def run_online_evaluation_3(self, tp_hard, fp_hard, fn_hard):
        tp_hard = tp_hard.detach().cpu().numpy().mean(0)
        fp_hard = fp_hard.detach().cpu().numpy().mean(0)
        fn_hard = fn_hard.detach().cpu().numpy().mean(0)
        self.online_eval_foreground_dc_3.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
        self.online_eval_tp_3.append(list(tp_hard))
        self.online_eval_fp_3.append(list(fp_hard))
        self.online_eval_fn_3.append(list(fn_hard))

    def process_plans(self, plans):
        super(HasTrainer_DP_DDense_3_branch, self).process_plans(plans)
        if not self.distribute_batch_size:

            self.batch_size = self.num_gpus * self.plans['plans_per_stage'][self.stage]['batch_size']
            # self.batch_size = self.plans['plans_per_stage'][self.stage]['batch_size']
            print("batch_size : ",self.batch_size)
            print("num_gpus: ",self.num_gpus)
            print("self.plans['plans_per_stage'][self.stage]['batch_size']: ", self.plans['plans_per_stage'][self.stage]['batch_size'])
            print("")

        else:

            if self.batch_size < self.num_gpus:
                print("WARNING: self.batch_size < self.num_gpus. Will not be able to use the GPUs well")
            elif self.batch_size % self.num_gpus != 0:
                print("WARNING: self.batch_size % self.num_gpus != 0. Will not be able to use the GPUs well")

    # def initialize(self, training=True, force_load_plans=False):
    #     """
    #     - replaced get_default_augmentation with get_moreDA_augmentation
    #     - only run this code once
    #     - loss function wrapper for deep supervision
    #
    #     :param training:
    #     :param force_load_plans:
    #     :return:
    #     """
    #     if not self.was_initialized:
    #         maybe_mkdir_p(self.output_folder)
    #
    #         if force_load_plans or (self.plans is None):
    #             self.load_plans_file()
    #
    #         self.process_plans(self.plans)
    #
    #         self.setup_DA_params()
    #
    #         ################# Here configure the loss for deep supervision ############
    #         net_numpool = len(self.net_num_pool_op_kernel_sizes)
    #         weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
    #         mask = np.array([True if i < net_numpool - 1 else False for i in range(net_numpool)])
    #         weights[~mask] = 0
    #         weights = weights / weights.sum()
    #         self.loss_weights = weights
    #         ################# END ###################
    #
    #         self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
    #                                                   "_stage%d" % self.stage)
    #         print("")
    #         print("folder_with_preprocessed_data : ", self.folder_with_preprocessed_data)
    #         print("plans : ", self.plans)
    #         print("labels : ", self.plans['all_classes'])
    #         print("")
    #
    #         if training:
    #             self.dl_tr, self.dl_val = self.get_basic_generators()
    #             if self.unpack_data:
    #                 print("unpacking dataset")
    #                 unpack_dataset(self.folder_with_preprocessed_data)
    #                 print("done")
    #             else:
    #                 print(
    #                     "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
    #                     "will wait all winter for your model to finish!")
    #
    #             self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
    #                                                                 self.data_aug_params[
    #                                                                     'patch_size_for_spatialtransform'],
    #                                                                 self.data_aug_params,
    #                                                                 deep_supervision_scales=self.deep_supervision_scales,
    #                                                                 pin_memory=self.pin_memory)
    #
    #             print("tr_gen : ", self.tr_gen)
    #             print("val_gen : ", self.val_gen)
    #
    #             # import pdb
    #             # pdb.set_trace()
    #
    #             self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
    #                                    also_print_to_console=False)
    #             self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
    #                                    also_print_to_console=False)
    #         else:
    #             pass
    #
    #         self.initialize_network()
    #         self.initialize_optimizer_and_scheduler()
    #
    #         assert isinstance(self.network, (SegmentationNetwork, DataParallel))
    #     else:
    #         self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
    #     self.was_initialized = True

    def initialize_network(self):
        if self.threeD:
            cfg = get_default_network_config(3, None, norm_type="in")

        else:
            cfg = get_default_network_config(1, None, norm_type="in")

        stage_plans = self.plans['plans_per_stage'][self.stage]
        conv_kernel_sizes = stage_plans['conv_kernel_sizes']
        blocks_per_stage_encoder = stage_plans['num_blocks_encoder']
        blocks_per_stage_decoder = stage_plans['num_blocks_decoder']
        pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        self.network = D_DenseUNet_Multi_Task_DP(self.num_input_channels, self.base_num_features, blocks_per_stage_encoder, 2,
                                      pool_op_kernel_sizes, conv_kernel_sizes, cfg, self.num_classes,
                                      blocks_per_stage_decoder, True, False, 320, InitWeights_He(1e-2))

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    # def initialize_optimizer_and_scheduler(self):
    #     assert self.network is not None, "self.initialize_network must be called first"
    #     self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
    #                                      momentum=0.99, nesterov=True)
    #     self.lr_scheduler = None



    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer_3_branch.validate(self, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                                     step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
                                     overwrite=overwrite, validation_folder_name=validation_folder_name,
                                     debug=debug, all_in_gpu=all_in_gpu,
                                     segmentation_export_kwargs=segmentation_export_kwargs,
                                     run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.decoder.deep_supervision = ds
        return ret



    # def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
    #                                                      mirror_axes: Tuple[int] = None,
    #                                                      use_sliding_window: bool = True, step_size: float = 0.5,
    #                                                      use_gaussian: bool = True,
    #                                                      pad_border_mode: str = 'constant',
    #                                                      pad_kwargs: dict = None, all_in_gpu: bool = False,
    #                                                      verbose: bool = True, mixed_precision=True) -> Tuple[
    #     np.ndarray, np.ndarray]:
    #     ds = self.network.decoder.deep_supervision
    #     self.network.decoder.deep_supervision = False
    #     ret = nnUNetTrainer_3_branch.predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=do_mirroring,
    #                                                                          mirror_axes=mirror_axes,
    #                                                                          use_sliding_window=use_sliding_window,
    #                                                                          step_size=step_size,
    #                                                                          use_gaussian=use_gaussian,
    #                                                                          pad_border_mode=pad_border_mode,
    #                                                                          pad_kwargs=pad_kwargs,
    #                                                                          all_in_gpu=all_in_gpu,
    #                                                                          verbose=verbose,
    #                                                                          mixed_precision=mixed_precision)
    #     self.network.decoder.deep_supervision = ds
    #     return ret
    #
    #
    #
    #
    # def predict_preprocessed_data_return_attention(self, data: np.ndarray, do_mirroring: bool = True,
    #                                                      mirror_axes: Tuple[int] = None,
    #                                                      use_sliding_window: bool = True, step_size: float = 0.5,
    #                                                      use_gaussian: bool = True,
    #                                                      pad_border_mode: str = 'constant',
    #                                                      pad_kwargs: dict = None, all_in_gpu: bool = False,
    #                                                      verbose: bool = True, mixed_precision=True) -> Tuple[
    #     np.ndarray, np.ndarray]:
    #     ds = self.network.decoder.deep_supervision
    #     self.network.decoder.deep_supervision = False
    #     ret = nnUNetTrainer_3_branch.predict_preprocessed_data_return_attention(self, data, do_mirroring=do_mirroring,
    #                                                                          mirror_axes=mirror_axes,
    #                                                                          use_sliding_window=use_sliding_window,
    #                                                                          step_size=step_size,
    #                                                                          use_gaussian=use_gaussian,
    #                                                                          pad_border_mode=pad_border_mode,
    #                                                                          pad_kwargs=pad_kwargs,
    #                                                                          all_in_gpu=all_in_gpu,
    #                                                                          verbose=verbose,
    #                                                                          mixed_precision=mixed_precision)
    #     self.network.decoder.deep_supervision = ds
    #     return ret








    # def run_training(self):
    #     self.maybe_update_lr(self.epoch)
    #     # self.maybe_update_lr()
    #
    #     #self.on_epoch_end()
    #
    #
    #     # amp must be initialized before DP
    #
    #     ds = self.network.do_ds
    #     self.network.do_ds = True
    #     self.network = DataParallel(self.network, tuple(range(self.num_gpus)), )
    #     # ret1, ret2, ret3 = nnUNetTrainer.run_training(self)
    #     ret = nnUNetTrainer_3_branch.run_training(self)
    #     self.network = self.network.module
    #     self.network.do_ds = ds
    #
    #     # ret = ret1 + ret2 + ret3
    #     return ret


    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder1.deep_supervision
        self.network.decoder1.deep_supervision = True

        # 이거 추가함 (멀티 gpu)
        # a = self.network
        self.network = DataParallel(self.network, tuple(range(self.num_gpus)), )

        ret = nnUNetTrainer_3_branch.run_training(self)

        ## if traning is end
        # print(self.epoch)
        # print(self.max_num_epochs)
        # if self.epoch == self.max_num_epochs - 1:
        #     self.network = a

        self.network.decoder.deep_supervision = ds



        return ret





    # 여기서 iteration이 돌아간다.
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']



        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        y1 = copy.deepcopy(target)
        y2 = copy.deepcopy(target)
        y3 = copy.deepcopy(target)



        # print("y1[0].max() : ", y1[0].max())
        # print("y2[0].max() : ", y2[0].max())
        # print("y3[0].max() : ", y3[0].max())


        # y1, y2, y3를 각각 1,2,3의 요소만 남기고 모두 0으로 만든다.
        for i in range(3):
            y1[i] = torch.where(y1[i] == 1, y1[i], torch.tensor(0).float())
            y2[i] = torch.where(y2[i] == 2, y2[i], torch.tensor(0).float())
            y3[i] = torch.where(y3[i] == 3, y3[i], torch.tensor(0).float())


        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            y1 = to_cuda(y1)
            y2 = to_cuda(y2)
            y3 = to_cuda(y3)


        # 역전파 단계를 실행하기 전에 gradient를 0으로 만든다.
        self.optimizer.zero_grad()

       # rint("optimizer : ", self.optimizer)


#       print("fp16 : ", self.fp16)

        # self.fp16 = True
        if self.fp16:
            with autocast():


                # return_hard_tp_fp_fn = run_online_evaluation = False 가 들어간다.
                ret1, ret2, ret3 = self.network(data, y1,y2,y3, return_hard_tp_fp_fn=run_online_evaluation)
             #  print("run_online_evaluation : ",run_online_evaluation)

                # # validation 할때 이리로 들어간다.
                # if run_online_evaluation:
                #     ces1, tps1, fps1, fns1, tp_hard1, fp_hard1, fn_hard1 = ret1
                #     self.run_online_evaluation(tp_hard1, fp_hard1, fn_hard1)
                #
                #     ces2, tps2, fps2, fns2, tp_hard2, fp_hard2, fn_hard2 = ret2
                #     self.run_online_evaluation(tp_hard2, fp_hard2, fn_hard2)
                #
                #     ces3, tps3, fps3, fns3, tp_hard3, fp_hard3, fn_hard3 = ret3
                #     self.run_online_evaluation(tp_hard3, fp_hard3, fn_hard3)


                # validation 할때 이리로 들어간다.
                if run_online_evaluation:
                    ces1, tps1, fps1, fns1, tp_hard1, fp_hard1, fn_hard1 = ret1
                    self.run_online_evaluation(tp_hard1, fp_hard1, fn_hard1)
                    # print("ces1 : ", ces1)
                    # print("tps1 : ", tps1)
                    # print("fps1 : ", fps1)
                    # print("tp_hard1 : ", tp_hard1)
                    # print("fp_hard1 : ", fp_hard1)
                    # print("fn_hard1 : ", fn_hard1)

                    ces2, tps2, fps2, fns2, tp_hard2, fp_hard2, fn_hard2 = ret2
                    self.run_online_evaluation_2(tp_hard2, fp_hard2, fn_hard2)

                    ces3, tps3, fps3, fns3, tp_hard3, fp_hard3, fn_hard3 = ret3
                    self.run_online_evaluation_3(tp_hard3, fp_hard3, fn_hard3)







                #   print("run_online_evaluation")




                # 이리로 들어간다
                else:
                    # CE_loss, TP, FP, FN을 여기서 받는다. (4개 채널로 구성된 tuple)

                    ces1, tps1, fps1, fns1 = ret1
                    ces2, tps2, fps2, fns2 = ret2
                    ces3, tps3, fps3, fns3 = ret3

                del data, target

                # 아래 compute_loss가 우리가 가장 신장하게 생각해야할 부분
                l1 = self.compute_loss(ces1, tps1, fps1, fns1)
                l2 = self.compute_loss(ces2, tps2, fps2, fns2)
                l3 = self.compute_loss(ces3, tps3, fps3, fns3)

                u = 0.7
                k = 0.15
                b = 0.15

                l = u*l1 + k*l2 + b*l3



            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()


        # 여기는 안쓴다고 보면 될듯 fp32 인 경우만 이리로 들어갈듯
        # else:
        #     ret = self.network(data, target, return_hard_tp_fp_fn=run_online_evaluation)
        #     if run_online_evaluation:
        #         ces, tps, fps, fns, tp_hard, fp_hard, fn_hard = ret
        #         self.run_online_evaluation(tp_hard, fp_hard, fn_hard)
        #     else:
        #         ces, tps, fps, fns = ret
        #     del data, target
        #     l = self.compute_loss(ces, tps, fps, fns)
        #
        #     if do_backprop:
        #         l.backward()
        #         torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        #         self.optimizer.step()

        return l.detach().cpu().numpy()



















    def compute_loss(self, ces, tps, fps, fns):
        # we now need to effectively reimplement the loss
        loss = None
        for i in range(len(ces)):
            if not self.dice_do_BG:
                tp = tps[i][:, 1:]
                fp = fps[i][:, 1:]
                fn = fns[i][:, 1:]
            else:
                tp = tps[i]
                fp = fps[i]
                fn = fns[i]

            if self.batch_dice:
                tp = tp.sum(0)
                fp = fp.sum(0)
                fn = fn.sum(0)
            else:
                pass

            nominator = 2 * tp + self.dice_smooth
            denominator = 2 * tp + fp + fn + self.dice_smooth

            dice_loss = (- nominator / denominator).mean()
            if loss is None:
                loss = self.loss_weights[i] * (ces[i].mean() + dice_loss)
            else:
                loss += self.loss_weights[i] * (ces[i].mean() + dice_loss)
        ###########
        return loss
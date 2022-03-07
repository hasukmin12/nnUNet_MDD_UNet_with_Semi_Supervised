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
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.generic_UNet_DP import Generic_UNet_DP
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.cuda.amp import autocast
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.utils import clip_grad_norm_


class nnUNetTrainerV2_DP(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, num_gpus=1, distribute_batch_size=False, fp16=False):
        super(nnUNetTrainerV2_DP, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                                unpack_data, deterministic, fp16)
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, num_gpus, distribute_batch_size, fp16)
        self.num_gpus = num_gpus
        self.distribute_batch_size = distribute_batch_size
        self.dice_smooth = 1e-5
        self.dice_do_BG = False
        self.loss = None
        self.loss_weights = None

        # self.batch_size = 4

    def setup_DA_params(self):
        super(nnUNetTrainerV2_DP, self).setup_DA_params()
        self.data_aug_params['num_threads'] = 8 * self.num_gpus

    def process_plans(self, plans):
        super(nnUNetTrainerV2_DP, self).process_plans(plans)
        if not self.distribute_batch_size:

            self.batch_size = self.num_gpus * self.plans['plans_per_stage'][self.stage]['batch_size']
            print("batch_size : ",self.batch_size)
            print("num_gpus: ",self.num_gpus)
            print("self.plans['plans_per_stage'][self.stage]['batch_size']: ", self.plans['plans_per_stage'][self.stage]['batch_size'])
            print("")

        else:

            if self.batch_size < self.num_gpus:
                print("WARNING: self.batch_size < self.num_gpus. Will not be able to use the GPUs well")
            elif self.batch_size % self.num_gpus != 0:
                print("WARNING: self.batch_size % self.num_gpus != 0. Will not be able to use the GPUs well")

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here configure the loss for deep supervision ############
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            mask = np.array([True if i < net_numpool - 1 else False for i in range(net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.loss_weights = weights
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            print("")
            print("folder_with_preprocessed_data : ", self.folder_with_preprocessed_data)
            print("plans : ", self.plans)
            print("labels : ", self.plans['all_classes'])
            print("")

            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                    self.data_aug_params[
                                                                        'patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    pin_memory=self.pin_memory)

                print("tr_gen : ", self.tr_gen)
                print("val_gen : ", self.val_gen)

                # import pdb
                # pdb.set_trace()

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        replace genericUNet with the implementation of above for super speeds
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet_DP(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_training(self):
        self.maybe_update_lr(self.epoch)
        # self.maybe_update_lr()

        #self.on_epoch_end()


        # amp must be initialized before DP

        ds = self.network.do_ds
        self.network.do_ds = True
        self.network = DataParallel(self.network, tuple(range(self.num_gpus)), )
        ret = nnUNetTrainer.run_training(self)
        self.network = self.network.module
        self.network.do_ds = ds
        return ret



    # 여기서 iteration이 돌아간다.
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        # print("data : ",data)
        # print("target : ", target)

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        # print("data : ", data)
        # print("target : ", target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        # 역전파 단계를 실행하기 전에 gradient를 0으로 만든다.
        self.optimizer.zero_grad()

       # rint("optimizer : ", self.optimizer)


#       print("fp16 : ", self.fp16)

        # self.fp16 = True
        if self.fp16:
            with autocast():

                # Training시에는 return_hard_tp_fp_fn = run_online_evaluation = False 가 들어간다.
                ret = self.network(data, target, return_hard_tp_fp_fn=run_online_evaluation)

             #  print("run_online_evaluation : ",run_online_evaluation)

                # validation 할때 이리로 들어간다.
                if run_online_evaluation:
                    ces, tps, fps, fns, tp_hard, fp_hard, fn_hard = ret
                    self.run_online_evaluation(tp_hard, fp_hard, fn_hard)
                #   print("run_online_evaluation")




                # Training시에는 이리로 들어간다
                else:
                    # CE_loss, TP, FP, FN을 여기서 받는다. (4개 채널로 구성된 tuple)
                #   print("run_online_evaluation is false")
                    ces, tps, fps, fns = ret

                del data, target

                # 아래 compute_loss가 우리가 가장 신장하게 생각해야할 부분
                l = self.compute_loss(ces, tps, fps, fns)

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



    def run_online_evaluation(self, tp_hard, fp_hard, fn_hard):
        tp_hard = tp_hard.detach().cpu().numpy().mean(0)
        fp_hard = fp_hard.detach().cpu().numpy().mean(0)
        fn_hard = fn_hard.detach().cpu().numpy().mean(0)
        self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
        self.online_eval_tp.append(list(tp_hard))
        self.online_eval_fp.append(list(fp_hard))
        self.online_eval_fn.append(list(fn_hard))

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
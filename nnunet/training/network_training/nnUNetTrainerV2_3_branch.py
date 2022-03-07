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


from collections import OrderedDict
from typing import Tuple
import copy
import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer_3_branch import nnUNetTrainer_3_branch
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


class nnUNetTrainerV2_3_branch(nnUNetTrainer_3_branch):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 300
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

        self.dice_smooth = 1e-5
        self.dice_do_BG = False
        self.loss = None
        self.loss_weights = None

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
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

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
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

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
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
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None


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

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret


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






    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        # 추가
        # target = target[0]
        # print("data : ",data)
        # print("target : ", target)

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        # print("data : ", data)
        # print("target[0].max() : ", target[0].max())

        y1 = copy.deepcopy(target)
        y2 = copy.deepcopy(target)
        y3 = copy.deepcopy(target)

        # print("before y1[0].max() : ", y1[0].max())
        # print("before y2[0].max() : ", y2[0].max())
        # print("before y3[0].max() : ", y3[0].max())

        # y1[0] = torch.where(y1[0]==1, y1[0], torch.tensor(0).float())

        # y1, y2, y3를 각각 1,2,3의 요소만 남기고 모두 0으로 만든다.
        for i in range(4):
            y1[i] = torch.where(y1[i] == 1, y1[i], torch.tensor(0).float())
            y2[i] = torch.where(y2[i] == 2, y2[i], torch.tensor(0).float())
            y3[i] = torch.where(y3[i] == 3, y3[i], torch.tensor(0).float())

        # print("y1[0].max() : ", y1[0].max())
        # print("y2[0].max() : ", y2[0].max())
        # print("y3[0].max() : ", y3[0].max())

        # if torch.cuda.is_available():
        #     data = to_cuda(data)
        #     y1 = to_cuda_3(y1)
        #     y2 = to_cuda_1(y2)
        #     y3 = to_cuda_2(y3)

        # 역전파 단계를 실행하기 전에 gradient를 0으로 만든다.
        self.optimizer.zero_grad()

        # rint("optimizer : ", self.optimizer)

        #       print("fp16 : ", self.fp16)

        # self.fp16 = True
        if self.fp16:
            with autocast():

                # return_hard_tp_fp_fn = run_online_evaluation = False 가 들어간다.
                ret1, ret2, ret3 = self.network(data, y1, y2, y3, return_hard_tp_fp_fn=run_online_evaluation)
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

                u = 0.8
                k = 0.1
                b = 0.1

                l = u * l1 + k * l2 + b * l3

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

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
    # def maybe_update_lr(self, epoch=300):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    # def run_training(self):
    #     """
    #     if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
    #     continued epoch with self.initial_lr
    #
    #     we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
    #     :return:
    #     """
    #     self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
    #     # want at the start of the training
    #
    #
    #
    #     ds = self.network.do_ds
    #     self.network.do_ds = True
    #     ret = super().run_training()
    #     self.network.do_ds = ds
    #     return ret

    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = True

        # 이거 추가함 (멀티 gpu)
        # self.network = DataParallel(self.network, tuple(range(self.num_gpus)), )

        ret = nnUNetTrainer_3_branch.run_training(self)
        self.network.decoder.deep_supervision = ds
        return ret

import argparse
import sys, os, time, yaml
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([os.path.dirname(os.path.dirname(os.path.abspath(__file__)))])
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_utils import TrainingProgress, timeSince, LearningRateScheduler, check_path, \
    ConfNamespace, ValueMeter, partial_load_weight
from multi_dataset import MultiDataset
from descriptor_evaluation import DescriptorEvaluation
from metric_loss import MetricLoss
from apex.fp16_utils import FP16_Optimizer
from shutil import copy2
import importlib

# from ipdb import set_trace
# import pydevd
#
# pydevd.settrace('140.113.216.12', port=9988, stdoutToServer=True, stderrToServer=True)
np.set_printoptions(precision=5)


class Trainer:
    def __init__(self, args, load_eval=True):
        self.load_config(args)  # args has higher priority that overwrites config file if value is specified

        if torch.cuda.is_available():
            torch.cuda.set_device(self.conf.dev)  # default 0
            print("Use CUDA,device=", torch.cuda.current_device())
            self.device = 'cuda:' + str(self.conf.dev)
        else:
            self.device = 'cpu'
        print('Config name', self.conf.exp_name)
        check_path(os.path.join(self.conf.result_path, self.conf.exp_name))
        copy2(args.config, os.path.join(self.conf.result_path, self.conf.exp_name))  # Config Backup

        self.tp = TrainingProgress(os.path.join(self.conf.progress_path, self.conf.exp_name), 'progress',
                                   data_key_list=['epoch_loss', 'test_loss', 'train_loss'])
        if args.draw:
            self.load_net()
            self.prepare_training()  # Optimizer/Loss function
            return
        self.load_net()
        if not args.test:
            self.load_data()
        if load_eval:
            self.load_evaluation()
        self.prepare_training()  # Optimizer/Loss function

    def load_evaluation(self, use_static=False, multi_static=False, load_pkl=False):
        """
        Static evaluation: control the random state so that same sampling config produce same result
        :param use_static:  Single class static evaluation
        :param multi_static: Multi-class static evalaution
        :param load_pkl: load
        :return:
        """
        self.evaluation = DescriptorEvaluation(mode='epoch', full_config=self.full_config, use_static=use_static,
                                               multi_static=multi_static, load_pkl=load_pkl)

    def load_net(self):
        """
        Load network weight or initiate a new model
        """
        FPN = importlib.import_module('models.' + self.model_conf.net_module)
        self.net = FPN.fpn50(self.model_conf, self.device)

        if not self.conf.res:
            if self.model_conf.load_FPN_pretrained:  # TODO: Never write shit code!
                print('Loading FPN pretrained weight.')
                dict_src = torch.load(self.model_conf.FPN_pretrained)
                dict_tgt = self.net.state_dict()
                dict_tgt = partial_load_weight(dict_src, dict_tgt)
                self.net.load_state_dict(dict_tgt)
        if self.model_conf.use_fp16:
            self.net = self.net.half()

        if self.model_conf.split_model:
            self.net.to_gpus()
        else:
            self.net = self.net.to(self.device)

        print('Trainable Parameter Count=', sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        print('Total Parameter Count=', sum(p.numel() for p in self.net.parameters()))

    def load_data(self):  # batch_files, data_conf, cuda_dev):
        """
        Load Training/Testing data
        """
        # print('Loading Dataset')

        self.train_loader = DataLoader(
            dataset=MultiDataset(self.full_config['sampling'], self.full_config['train_data']),
            batch_size=self.conf.tr_bat, shuffle=False,
            pin_memory=True, num_workers=self.conf.worker)
        print('Data loader: Training set: ', len(self.train_loader.dataset))

    def prepare_training(self):
        """
        Load/Create Meta data
        Load/Restore Current Progress
        Set training parameters, Init optimizer
        """
        self.tp.add_meta({'training_conf': self.conf, 'criterion_conf': self.full_config['criterion'],
                          'model_conf': self.model_conf})
        if self.conf.res:
            self.restore_progress()  # set optimizer and lr_scheduler
        else:
            self.epoch = 1
            self.set_optimizer()
            self.set_lr_scheduler()
        self.loss_eval = MetricLoss(ConfNamespace(self.full_config['criterion']), self.device)

        # self.evaluation = Evaluation(self.full_config, self.device)
        # self.evaluation.random_get_testing_set()

    def train(self):
        """
        while loss<target loss
            forward
            backward
            record loss
            if loop_n % RECORD_N:
                summary & save_progress
        """
        time_start = time.time()
        self.net.train()
        loss_recorder = ValueMeter()
        # For tracking centers
        # try:
        device_img = self.model_conf.gpu_seq[0]

        while self.epoch <= self.conf.max_epoch:  # self.epoch start from 1
            self.epoch_loss = 0
            loss_recorder.reset()
            for i, values in enumerate(self.train_loader):
                """
                values is tuple returned from the dataloader
                elements 0,1,2 are the normalized input images
                other elements are unpacked in the loss evaluation stage
                """
                self.optimizer.zero_grad()
                if self.loss_conf.A_A2_only:
                    if self.model_conf.split_model:
                        img_A = values[0].to(device_img)
                        img_A2 = values[1].to(device_img)
                    else:
                        img_A = values[0].to(self.device)
                        img_A2 = values[1].to(self.device)

                    if self.model_conf.use_fp16:
                        img_A, img_A2 = tuple(map(lambda x: x.half(), [img_A, img_A2]))
                    desp_A = self.net(img_A).float()
                    desp_A2 = self.net(img_A2).float()
                    desp_B = None
                else:
                    if self.model_conf.split_model:
                        img_A = values[0].to(device_img)
                        img_A2 = values[1].to(device_img)
                        img_B = values[2].to(device_img)
                    else:
                        img_A = values[0].to(self.device)
                        img_A2 = values[1].to(self.device)
                        img_B = values[2].to(self.device)

                    if self.model_conf.use_fp16:
                        img_A, img_A2, img_B = tuple(map(lambda x: x.half(), [img_A, img_A2, img_B]))

                    # [1,3,480,640]
                    desp_A = self.net(img_A).float()
                    desp_A2 = self.net(img_A2).float()
                    desp_B = self.net(img_B).float()

                self.loss_eval.set_data(desp_A, desp_A2, desp_B, values[3:])
                self.loss_eval.set_centers(self.net.centers.float())
                if self.loss_conf.split_match:
                    loss_obj_match, loss_obj_bk, loss_bk_bk = self.loss_eval.get_matching_loss(split=True)
                    loss_match = self.loss_conf.match_alpha_obj * loss_obj_match + \
                                 self.loss_conf.non_match_alpha_obj_bk * loss_obj_bk + \
                                 self.loss_conf.match_alpha_bk_bk * loss_bk_bk
                else:
                    loss_match = self.loss_conf.match_alpha * self.loss_eval.get_matching_loss()
                loss_hard = self.loss_conf.hard_alpha * self.loss_eval.get_hard_negative_loss()
                total_loss = loss_match + loss_hard
                if self.model_conf.use_class:
                    if self.loss_conf.use_hard_metric:
                        loss_class = self.loss_conf.clf_alpha * self.loss_eval.get_hard_metric_loss(self.epoch)
                    else:
                        loss_class = self.loss_conf.clf_alpha * self.loss_eval.get_discriminative_loss(self.net,
                                                                                                       self.model_conf.use_fp16)
                    total_loss += loss_class

                if not self.loss_conf.no_triplet:
                    loss_triplet = self.loss_conf.triplet_alpha * self.loss_eval.get_triplet_center_loss()
                    total_loss += loss_triplet

                if self.model_conf.learn_center:
                    centers_loss = self.loss_conf.center_alpha * self.loss_eval.get_loss_centers()

                norm_loss = self.loss_conf.loss_norm_alpha * self.loss_eval.get_space_normalization_loss()
                total_loss += norm_loss

                if self.model_conf.use_fp16:
                    self.optimizer.backward(total_loss)
                    if self.model_conf.learn_center:
                        self.optimizer.backward(centers_loss)
                        total_loss += centers_loss
                else:
                    total_loss.backward()
                    if self.model_conf.learn_center:
                        centers_loss.backward()
                        total_loss += centers_loss
                self.optimizer.step()

                self.epoch_loss += total_loss.item()  # + centers_loss.item()
                pix_map = self.loss_eval.get_match_distance()  # Not used, but for loss_recorder

                loss_recorder.record_data(self.loss_eval.get_loss_records())
                # print('Loss recorder',self.loss_eval.get_loss_records())
                if (i + 1) % 10 == 0:
                    print('Data ', i + 1, ' of ', len(self.train_loader.dataset))
                # print(self.loss_eval.get_loss_records())
            self.epoch_loss = self.epoch_loss / len(self.train_loader.dataset)
            self.tp.record_epoch(self.epoch, 'train',
                                 {'epoch_loss': self.epoch_loss},
                                 display=True)  # 'validation_loss': self.valid_loss})
            self.tp.record_epoch(self.epoch, 'train', loss_recorder.avg(), display=True)
            self.lr_scheduler.step(
                {'loss': self.epoch_loss, 'epoch': self.epoch, 'decay': self.epoch})  # , 'torch': self.valid_loss})
            if self.model_conf.learn_center:
                self.center_lr_scheduler.step({'epoch': self.epoch, 'decay': self.epoch})
            print('Current centers:')
            self.net.centers.print_centers()
            self.evaluation_desp()
            if self.epoch % self.conf.se == 0:
                print(timeSince(time_start), ': Trainer Summary Epoch=', self.epoch)
                if self.epoch < self.conf.max_epoch:
                    self.summary()
                else:
                    # self.summary(save_optim=True)  # for resume training
                    self.summary(save_optim=False)  # TODO: Check optimizer state dict when using apex

                # print('Current Centers=\n', self.net.centers.detach().cpu().numpy())
                self.tp.plot_epoch_data('train', 1, self.epoch, \
                                        os.path.join(os.path.join(self.conf.result_path, self.conf.exp_name),
                                                     'train_loss.png'), 'Training Loss')
                self.tp.plot_epoch_data('test', 1, self.epoch, \
                                        os.path.join(os.path.join(self.conf.result_path, self.conf.exp_name),
                                                     'test_loss.png'), 'Testing Loss')
            else:
                print(timeSince(time_start), ': Trainer Epoch=', self.epoch)
            # Some more things...
            # adjust alpha ???
            self.epoch += 1
            if self.device != 'cpu':
                torch.cuda.empty_cache()

        # except KeyboardInterrupt:
        #     save = input('Save Current Progress ? y for yes: ')
        #     if 'y' in save:
        #         print('Saving Progress...')
        #         self.save_progress(save_optim=True, display=True)

    def set_optimizer(self):
        """
        Set optimizer parameters
        """
        if not self.model_conf.learn_center:
            if self.conf.optim == 'SGD':
                self.optimizer = getattr(optim, 'SGD')(filter(lambda p: p.requires_grad, self.net.parameters()),
                                                       lr=self.conf.lr_init, momentum=0.9, nesterov=True,
                                                       weight_decay=self.conf.w_decay)  # default SGD
            else:
                self.optimizer = getattr(optim, self.conf.optim)(
                    filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.conf.lr_init,
                    weight_decay=self.conf.w_decay)  # default SGD
        else:  # Learn center
            params_model = []
            params_center = []
            for n, p in self.net.named_parameters():
                if 'centers' in n and p.requires_grad:  # TODO: check if classifier is also better if separated
                    params_center.append(p)
                elif p.requires_grad:
                    params_model.append(p)

            if self.conf.optim == 'SGD':
                self.optimizer = getattr(optim, 'SGD')(params_model, lr=self.conf.lr_init, momentum=0.9, nesterov=True,
                                                       weight_decay=self.conf.w_decay)
                self.optimizer.add_param_group({'params': params_center})
            else:
                self.optimizer = getattr(optim, self.conf.optim)(params_model, lr=self.conf.lr_init,
                                                                 weight_decay=self.conf.w_decay)
                self.optimizer.add_param_group(
                    {'params': params_center})  # Other settings are same as the first group by default

        if self.model_conf.use_fp16:
            self.optimizer = FP16_Optimizer(self.optimizer)
        if self.conf.res:
            if self.tp.get_meta('optim') == self.conf.optim:
                if 'optim_state' in self.tp.meta_dict.keys():
                    self.optimizer.load_state_dict(self.tp.get_meta('optim_state'))
                    print('Optimizer Internal State Restored')
        # Test

    def set_lr_scheduler(self, restore_dict=None):
        if self.conf.lrs == 'decay':
            self.lr_scheduler = LearningRateScheduler(self.conf.lrs, [self.optimizer.param_groups[0]],
                                                      self.conf.lr_rates,
                                                      self.conf.lr_epochs, self.conf.lr_loss, self.conf.lr_init,
                                                      lambda x: self.conf.lr_decay_factor)
            if self.model_conf.learn_center:
                self.center_lr_scheduler = LearningRateScheduler(self.conf.ct_lrs, [self.optimizer.param_groups[1]],
                                                                 self.conf.ct_lr_rates,
                                                                 self.conf.ct_lr_epochs, self.conf.ct_lr_loss,
                                                                 self.conf.ct_lr_init,
                                                                 lambda x: self.conf.ct_lr_decay_factor)
        else:
            self.lr_scheduler = LearningRateScheduler(self.conf.lrs, [self.optimizer.param_groups[0]],
                                                      self.conf.lr_rates,
                                                      self.conf.lr_epochs, self.conf.lr_loss, self.conf.lr_init, None)
            if self.model_conf.learn_center:
                self.center_lr_scheduler = LearningRateScheduler(self.conf.ct_lrs, [self.optimizer.param_groups[1]],
                                                                 self.conf.ct_lr_rates,
                                                                 self.conf.ct_lr_epochs, self.conf.ct_lr_loss,
                                                                 self.conf.ct_lr_init,
                                                                 lambda x: self.conf.ct_lr_decay_factor)

        if restore_dict is not None:
            Warning('TODO: Center LR not restored')
            if self.conf.lrs == 'epoch':
                self.lr_scheduler.step(restore_dict)
            elif self.conf.lrs == 'decay':
                restore_dict.update({'decay': restore_dict['epoch']})
                self.lr_scheduler.lr_rates = self.conf.lr_init * (self.conf.lr_decay_factor ** restore_dict['epoch'])
                self.lr_scheduler.step(restore_dict)
                if self.model_conf.learn_center:
                    self.center_lr_scheduler.lr_rates = self.conf.ct_lr_init * (
                            self.conf.ct_lr_decay_factor ** restore_dict['epoch'])
                    self.center_lr_scheduler.step(restore_dict)

    def summary(self, save_optim=False, save_progress=True):  # Do the tests
        """
        Record the training and testing loss/time/accuracy
        """
        # train_loss = self.test(use_training=True, display=True)
        # test_loss = self.test(display=True)
        # self.tp.record_epoch(self.epoch, 'test', {'test_loss': test_loss}, display=True)

        self.tp.add_meta(
            {'saved_epoch': self.epoch, 'epoch_loss': self.epoch_loss})  # , 'validation_loss': self.valid_loss})
        if save_progress:
            self.save_progress(display=True, save_optim=save_optim)

    def save_progress(self, display=False, save_optim=False):
        """
        Save training weight/progress/meta data
        """
        self.tp.add_meta(
            {'net_weight': {k: v.cpu() for k, v in self.net.state_dict().items()}, 'optim': self.conf.optim})
        if self.conf.save_opt or save_optim:
            print('Saving Optimizer Sate')
            self.tp.add_meta({'optim_state': self.optimizer.state_dict()})
        self.tp.save_progress(self.epoch)
        # self.net = self.net.to(self.device)
        if display:
            print('Config name', self.conf.exp_name)
            print('Progress Saved, current epoch=', self.epoch)

    def restore_progress(self):
        """
        Restore training weight/progress/meta data
        Restore self.epoch,optimizer parameters
        """
        self.tp.restore_progress(self.conf.tps)

        self.net = self.net.to('cpu')
        if self.model_conf.use_fp16:
            self.net = self.net.half()

        self.net.load_state_dict(self.tp.get_meta('net_weight'))

        # self.net = self.net.to(self.device)
        if self.model_conf.split_model:
            self.net.to_gpus()
        else:
            self.net = self.net.to(self.device)
        # restore all the meta data and variables
        self.epoch = self.tp.get_meta('saved_epoch')
        self.epoch_loss = self.tp.get_meta('epoch_loss')
        # self.valid_loss = self.tp.get_meta('validation_loss')
        print('Restore Progress,epoch=', self.epoch, ' epoch loss=', self.epoch_loss)
        self.set_optimizer()
        self.set_lr_scheduler(restore_dict={'epoch': self.epoch})
        self.epoch += 1  # next epoch

    def evaluation_desp(self, plot_desp=True, use_static=False, pixd_use_class=False):
        # TODO: Add an evaluation loss recorder and save to progress as test loss
        if use_static:
            avg_eval_loss, img_pairs, desp_pairs, loss_maps, pck_result, normalized_pck_result = self.evaluation.run_training_evaluation(
                self.net,
                self.loss_eval,
                plot_desp, pixd_use_class)
        else:
            avg_eval_loss, img_pairs, desp_pairs, loss_maps, pck_result = self.evaluation.run_training_evaluation(
                self.net,
                self.loss_eval,
                plot_desp, pixd_use_class)

        if plot_desp:
            self.evaluation.plot_visualization(os.path.join(self.conf.result_path, self.conf.exp_name), self.epoch,
                                               img_pairs, desp_pairs)
            self.evaluation.plot_cdf(os.path.join(self.conf.result_path, self.conf.exp_name), self.epoch, loss_maps)

        self.tp.record_epoch(self.epoch, 'test', avg_eval_loss, display=True)
        self.tp.record_epoch(self.epoch, 'test', pck_result, display=True)
        if use_static:  # Only in testing
            # self.tp.record_epoch(self.epoch, 'test', normalized_pck_result, display=False)
            return pck_result, normalized_pck_result, avg_eval_loss['class_accuracy']
        return pck_result, avg_eval_loss['class_accuracy']

    def load_config(self, args):
        # Load yaml configs
        # Command line arguments override the yaml configs if provided
        with open(args.config, 'r') as stream:
            self.full_config = yaml.load(stream)  # Full config for reference

        # the keys of "full_config" is the first layer key in the config yaml file
        if args.overwrite_test is not None:
            with open(args.overwrite_test, 'r') as stream:
                overwrite_test_config = yaml.load(stream)  # Full config for reference
            self.full_config['sampling'] = overwrite_test_config['sampling']
            self.full_config['test_data'] = overwrite_test_config['test_data']
            self.full_config['evaluation'] = overwrite_test_config['evaluation']
            print('Trainer: Overwrite sampling/test data/evaluation config ')
        if args.test_set_n is not None:
            self.full_config['evaluation']['test_set_n'] = args.test_set_n
        self.conf = ConfNamespace(self.full_config['training_param'], args.__dict__)
        self.model_conf = ConfNamespace(self.full_config['model'])
        self.loss_conf = ConfNamespace(self.full_config['criterion'])
        self.eval_conf = ConfNamespace(self.full_config['evaluation'])

    # def adjust_alpha(self):
    #     self.desp_param.alpha = np.clip(self.desp_param.alpha + 0.03, self.desp_param.alpha,
    #                                     self.desp_param.alpha_upper)
    #
    # def adjust_pixel_threshold(self):
    #     self.desp_param.pixel_threshold = np.clip(self.desp_param.pixel_threshold - 1, 0,
    #                                               self.desp_param.pixel_threshold)
    #

    def run_single_static_evaluation(self):
        """
        Override trainer's evaluation_desp
        Hack args that is sent to descriptor_evaluation
        Net is kept, but create new descriptor_evaluation for different class settings
        """
        s_cf = ConfNamespace(self.full_config['test_single'])
        self.full_config['criterion']['A_A2_only'] = True  # For Descriptor Evaluation Init
        self.full_config['criterion']['hard_negative_only_self'] = False  # Only single class
        self.loss_conf.A_A2_only = True  # For trainer
        # self.full_config['evaluation']['test_set_n'] = 5  # Testing Purpose
        # Reset Metric Loss Setting
        self.loss_eval = MetricLoss(ConfNamespace(self.full_config['criterion']), self.device)
        self.full_config['sampling'].update({  # Use same sampling params
            'pix_dist_match_n': 300,
            'hard_negative_match': 350,
            'hard_negative_AB_n': 300,
            'mask_max_hard_negative': 80000,
            'scale_range_min': 0.7,
            'scale_range_max': 1.0
        })
        self.full_config['evaluation']['test_set_n'] = -1

        result_dict = {}
        if len(self.full_config['test_single']['total_classes']) == 1:  # Single class model
            self.full_config['test_data']['object_classes'] = self.full_config['test_single']['total_classes']
        for frame_near in s_cf.frame_near:
            for cls in s_cf.total_classes:
                self.full_config['test_data']['match_classes'] = [cls]
                self.full_config['sampling']['match_frame_near'] = frame_near
                print('Sampling for FrameNear={},Class={}'.format(frame_near, cls))
                self.load_evaluation(use_static=True)
                pck_result, normalized_pck_result, clf_result = self.evaluation_desp(plot_desp=False, use_static=True)
                result_dict['Class:{}, MFN:{}'.format(cls, frame_near)] = pck_result
                result_dict['Class:{}, MFN:{} Normalized'.format(cls, frame_near)] = normalized_pck_result
                result_dict['Class:{}, MFN:{},accuracy'.format(cls, frame_near)] = clf_result
                print(pck_result, normalized_pck_result)
        for k, v in result_dict.items():
            result_dict[k] = {kr: vr.tolist() for kr, vr in v.items()} if isinstance(v, dict) else v.tolist()
        with open(os.path.join(self.conf.result_path, self.conf.exp_name, s_cf.result_name), 'w') as f:
            yaml.dump(result_dict, f, default_flow_style=False)

    def run_multi_static_evaluation(self, pixd_class=False, use_train=False, load_pkl=False):
        """
        Override trainer's evaluation_desp
        Hack args that is sent to descriptor_evaluation
        Net is kept, but create new descriptor_evaluation for different class settings
        """
        if use_train:
            self.full_config['test_data'] = self.full_config['train_data']
        s_cf = ConfNamespace(self.full_config['test_single'])
        self.full_config['criterion']['A_A2_only'] = True  # For Descriptor Evaluation Init
        self.loss_conf.A_A2_only = True  # For trainer
        # self.full_config['evaluation']['test_set_n'] = 5  # Testing Purpose
        # Reset Metric Loss Setting
        self.loss_eval = MetricLoss(ConfNamespace(self.full_config['criterion']), self.device)
        self.full_config['sampling'].update({  # Use same sampling params to produce same result
            'pix_dist_match_n': 300,
            'hard_negative_match': 350,
            'hard_negative_AB_n': 300,
            'mask_max_hard_negative': 80000,
            'matching_class_set_n': 4,
            'non_matching_class_set_n': 3,
            'scale_range_min': 0.7,
            'scale_range_max': 1.0
        })
        self.full_config['evaluation']['test_set_n'] = 230  # 230
        self.full_config['test_data']['sample_len'] = 230  # 230
        self.full_config['evaluation']['pck_pix_list'] = [40, 80, 120, 160]

        result_dict = {}
        for frame_near in s_cf.frame_near:
            self.full_config['sampling']['match_frame_near'] = frame_near
            print('Sampling for FrameNear={}'.format(frame_near))
            self.load_evaluation(use_static=False, multi_static=True, load_pkl=load_pkl)
            pck_result, clf_result = self.evaluation_desp(plot_desp=True, pixd_use_class=pixd_class)
            result_dict['MFN:{}'.format(frame_near)] = pck_result
            result_dict['MFN:{},accuracy'.format(frame_near)] = clf_result
            print(pck_result)
        for k, v in result_dict.items():
            result_dict[k] = {kr: vr.tolist() for kr, vr in v.items()} if isinstance(v, dict) else v.tolist()
        with open(os.path.join(self.conf.result_path, self.conf.exp_name, 'multi-' + s_cf.result_name), 'w') as f:
            yaml.dump(result_dict, f, default_flow_style=False)


def main():
    global cf, Net
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Training Config file is required", type=str)

    parser.add_argument('-model', help='Model file name', type=str)
    parser.add_argument('-opt_header', help='Additional weight/progress file header', type=str)
    parser.add_argument('-dev', help='CUDA Device Number, used when GPU is available', type=int)
    parser.add_argument('-worker', help='Dataloader Worker Number', type=int)

    # Learning rate scheduler setting
    parser.add_argument('-lrs', help='Learning rate scheduler mode', type=str)
    parser.add_argument('-lr_init', help='Initial Learning rate for Torch lr Schedulers', type=float)
    parser.add_argument('-lr_rates', help='Learning Rates for epochs.', nargs='+', type=float)
    parser.add_argument('-lr_epochs', help='Epochs for Learning Rate control.', nargs='+', type=int)
    parser.add_argument('-lr_loss', help='Loss targets for Learning Rate control.', nargs='+', type=float)

    parser.add_argument('-optim', help='Overwrite the optimizer, default=SGD.',
                        choices=['RMSprop', 'SGD', 'Adadelta', 'Adam'])
    parser.add_argument('-w_decay', help='Weight decay parameter for Optimizer. Ex: 1e-5', type=float)

    parser.add_argument('-se', help='Save progress and do summary every ? epoch', type=int)
    parser.add_argument('--res', help='Restore progress and resume the training progress',
                        action='store_true')
    parser.add_argument('-tps', help='Restore Training Progress index(step)', type=int)
    parser.add_argument('--save_opt', help='Save optimizer State Dict (Take some time !)?', action='store_true')

    parser.add_argument('--test', help='Display Testing Result only!', action='store_true')

    parser.add_argument('-tr_bat', help='Training Batch Size', type=int)
    parser.add_argument('-ts_bat', help='Testing Batch Size', type=int)

    parser.add_argument('-max_epoch', help='Max Epoch for training', type=int)
    parser.add_argument('--init_fin', help='Weight Initialization by the fan-in size', action='store_true')
    parser.add_argument('--draw', help='Draw saved epoch data', action='store_true', default=False)
    parser.add_argument('-draw_epoch', help='Epoch for drawing', default=0, type=int)
    parser.add_argument('-test_set_n', help='Evaluation Testing set N', type=int, default=None)
    parser.add_argument('-overwrite_test', help='Override Sampling and Folder config', type=str, default=None)
    parser.add_argument('--run_single', help='Run Single Static Test for each class', action='store_true')
    parser.add_argument('--run_multi', help='Run Multi Static Test', action='store_true')
    parser.add_argument('--use_train', help='Run Multi Static Test using training data', action='store_true')
    parser.add_argument('--pixd_class', help='Evaluate pixel distance using classification result', action='store_true')
    parser.add_argument('--pkl', help='Load Multi Static 7 class data from disk', action='store_true', default=False)
    parser.add_argument('--eval', help='Eval mode', action='store_true', default=False)
    args = parser.parse_args()
    if args.test:
        # assert args.res
        if args.run_single:
            trainer = Trainer(args, load_eval=False)
            del trainer.optimizer
            gc.collect()
            torch.cuda.empty_cache()
            trainer.run_single_static_evaluation()
        elif args.run_multi:
            trainer = Trainer(args, load_eval=False)
            del trainer.optimizer
            gc.collect()
            torch.cuda.empty_cache()
            trainer.run_multi_static_evaluation(args.pixd_class, args.use_train, args.pkl)
        else:
            trainer = Trainer(args)
            del trainer.optimizer
            gc.collect()
            torch.cuda.empty_cache()
            trainer.evaluation_desp(pixd_use_class=args.pixd_class)
    elif args.draw:
        trainer = Trainer(args)
        del trainer.optimizer
        gc.collect()
        torch.cuda.empty_cache()
        trainer.tp.plot_epoch_data('train', 1, args.draw_epoch, \
                                   os.path.join(os.path.join(trainer.conf.result_path, trainer.conf.exp_name),
                                                'train_loss_plot{}.png'.format(args.draw_epoch)), 'Training Loss')
        trainer.tp.plot_epoch_data('test', 1, args.draw_epoch, \
                                   os.path.join(os.path.join(trainer.conf.result_path, trainer.conf.exp_name),
                                                'test_loss-plot{}.png'.format(args.draw_epoch)), 'Testing Loss')
    else:
        trainer = Trainer(args)
        trainer.train()


if __name__ == '__main__':
    main()

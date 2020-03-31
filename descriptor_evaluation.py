import numpy as np
import torch, importlib
from torchvision import transforms as tf
import os, gc
from sklearn.decomposition import PCA
from PIL import Image
import cv2
import argparse
from utils.training_utils import TrainingProgress, ConfNamespace, ValueMeter
from multi_dataset import MultiDataset
from static_dataset import StaticDataset
from visual_dataset import VisualDataset
import yaml
import torch.backends.cudnn as cudnn
import pickle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

colors = {'red': (0, 0, 255),  # BGR
          'green': (0, 255, 0),
          'blue': (255, 0, 0),
          'black': (0, 0, 0),
          'yellow': (0, 255, 255)}

os.putenv('DISPLAY', ':0')


class DescriptorEvaluation:
    # def __init__(self, path, img_a, img_b, device='cuda:0'):
    def __init__(self, args=None, mode='visual', full_config=None, use_static=False,
                 multi_static=False, load_pkl=False):  # only for args.config
        self.use_static = use_static
        self.multi_static = multi_static
        if mode == 'visual':
            assert args is not None
            self.load_config(args)
        else:
            assert full_config is not None
            self.full_config = full_config
            self.conf = ConfNamespace(full_config['evaluation'])
            print('Test set n=', self.conf.test_set_n)
            self.model_conf = ConfNamespace(full_config['model'])
            self.tr_conf = ConfNamespace(full_config['training_param'])
            self.loss_conf = ConfNamespace(full_config['criterion'])

        if use_static:
            self.loss_conf.A_A2_only = True  # Static Dataset A A2 Only
            self.dataset = StaticDataset(self.full_config['sampling'], self.full_config['test_data'])
        else:
            self.dataset = MultiDataset(self.full_config['sampling'], self.full_config['test_data'], multi_static)

        # TODO: clean up the logic
        if mode == 'visual':
            if args.eval:
                self.dataset = VisualDataset(self.full_config['visual_data'])

        if mode == 'visual':  # Visual evaluation
            self.tp = TrainingProgress(os.path.join(self.tr_conf.progress_path, self.tr_conf.exp_name), 'progress')
            if args.tps is not None:
                restore_tps = args.tps
            else:
                restore_tps = self.tr_conf.tps
            print('Restoring progress ', restore_tps)
            self.tp.restore_progress(restore_tps)
            if torch.cuda.is_available():
                torch.cuda.set_device(self.conf.device)  # default 0
                print("Evaluation use CUDA,device=", torch.cuda.current_device())
                self.device = 'cuda:' + str(self.conf.device)
                cudnn.benchmark = True
            else:
                self.device = 'cpu'
            FPN = importlib.import_module('models.' + self.model_conf.net_module)
            self.net = FPN.fpn50(self.model_conf, self.device)
            self.net = self.net.to('cpu')
            if self.model_conf.use_fp16:
                self.net = self.net.half()
            self.net.load_state_dict(self.tp.get_meta('net_weight'))
            # self.net = self.net.to(self.device)
            if self.model_conf.split_model:
                self.net.to_gpus()
            self.net = self.net.to(self.device)
            self.net.eval()
            self.transform = tf.Compose([  # Should be same to the Dataset
                tf.ToTensor(),
                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                lambda x: x.to(self.device)  # cuda()
            ])
            # self.dataset.match_frame_near = self.dataset.match_frame_near *2  # larger range sampling
        else:  # 'epoch' Evaluation during training epochs
            self.test_data = []
            no_jit_transform = tf.Compose([
                tf.ToTensor(),
                tf.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])
            self.dataset.transform = no_jit_transform
            if not use_static:
                rsr_backup = self.dataset.random_scale_rotate
                self.dataset.random_scale_rotate = False
            print('Evaluation: Sampling Testing Data...')
            if self.conf.test_set_n == -1:
                self.conf.test_set_n = len(self.dataset)
                print('Use full dataset length={} for testing'.format(len(self.dataset)))

            # for i in range(self.conf.test_set_n):
            #     values = list(self.dataset[0])
            #     make_batch = tuple(
            #         [torch.tensor(i[None]) if isinstance(i, np.ndarray) else i[None].clone() for i in values])
            #     self.test_data.append(make_batch)
            if use_static:
                self.dataset.sample_len = min(self.conf.test_set_n, self.dataset.sample_len)
            else:
                self.dataset.sample_len = self.conf.test_set_n
            print('Final Testing Data Length=', len(self.dataset))
            if multi_static and load_pkl:
                print('Loading Multi Static Testing File: multi_static_10_7cls.pkl')
                with open('multi_static_10_7cls.pkl', 'rb') as f:
                    self.test_data = pickle.load(f)
                print('multi_static_10_7cls Loaded')
            else:
                tmp_loader = DataLoader(self.dataset, batch_size=1, num_workers=16)
                for i, v in enumerate(tmp_loader):
                    self.test_data.append(v)
            print('Done!')
            if not use_static:
                self.dataset.random_scale_rotate = rsr_backup

    """
    Functions for visual evaluation 
    """

    def fwd(self, x):
        if self.model_conf.use_fp16:
            return self.net(self.transform(x)[None].half())[0].float()
        else:
            return self.net(self.transform(x)[None])[0]

    def clf(self, desp_2d):
        if self.model_conf.use_fp16:
            return self.net.clf(desp_2d.view(desp_2d.size(0), -1).permute(1, 0).half()).float()
        else:
            return self.net.clf(desp_2d.view(desp_2d.size(0), -1).permute(1, 0))

    def sample_testing_images(self):
        self.dataset.transform = lambda x: x  # Cancel normalization
        values = self.dataset[0]
        img_A, img_B = values[0], values[1]
        # 480,640,3  RGB format, np.float64 ,[0,1]
        self.rgb_A = img_A.astype(np.float32) / 255.0
        self.rgb_B = img_B.astype(np.float32) / 255.0

        # Forward Result one time
        print('Forwarding Images...')
        self.net.eval()
        height, width = self.rgb_A.shape[0], self.rgb_A.shape[1]
        assert self.model_conf.use_class
        with torch.no_grad():
            desp_A, desp_B = self.fwd(self.rgb_A.copy()), self.fwd(self.rgb_B.copy())
            label_A, label_B = torch.max(self.clf(desp_A.detach()), 1)[1], \
                               torch.max(self.clf(desp_B.detach()), 1)[1]  # 307200,1
        self.label_A = label_A.cpu().numpy().reshape((height, width))
        self.label_B = label_B.cpu().numpy().reshape((height, width))
        self.desp_A = np.moveaxis(desp_A.cpu().numpy(), 0, 2)
        self.desp_B = np.moveaxis(desp_B.cpu().numpy(), 0, 2)
        self.draw_mask_from_label()
        print('Done!')
        # For Visualization
        self.desp_A_vis = self.desp_A.copy()
        self.desp_B_vis = self.desp_B.copy()
        if self.use_pca:
            self.desp_A_vis, self.desp_B_vis = self.remap_descriptor_pair(self.desp_A_vis, self.desp_B_vis)
        else:
            self.desp_A_vis = self.remap_descriptor_vis(self.desp_A_vis)
            self.desp_B_vis = self.remap_descriptor_vis(self.desp_B_vis)

    def specify_testing_images(self, path_A, path_B):  # Remain for convenience, Legacy
        self.rgb_A = np.asarray(Image.open(path_A)).astype(np.float32) / 255.0
        self.rgb_B = np.asarray(Image.open(path_B)).astype(np.float32) / 255.0
        # Forward Result one time
        print('Forwarding Images...')
        self.net.eval()
        height, width = self.rgb_A.shape[0], self.rgb_A.shape[1]
        # self.fwd = lambda x: np.moveaxis(self.net(self.transform(x)[None])[0].cpu().numpy(), 0,2)  # 480,640,3,float32
        with torch.no_grad():
            desp_A = self.fwd(self.rgb_A.copy())
            desp_B = self.fwd(self.rgb_B.copy())
            label_A, label_B = torch.max(self.clf(desp_A.detach()), 1)[1], \
                               torch.max(self.clf(desp_B.detach()), 1)[1]  # 307200,1
        self.label_A = label_A.cpu().numpy().reshape((height, width))
        self.label_B = label_B.cpu().numpy().reshape((height, width))
        self.desp_A = np.moveaxis(desp_A.cpu().numpy(), 0, 2)
        self.desp_B = np.moveaxis(desp_B.cpu().numpy(), 0, 2)
        self.draw_mask_from_label()
        # For Visualization
        self.desp_A_vis = self.desp_A.copy()
        self.desp_B_vis = self.desp_B.copy()
        if self.use_pca:
            self.desp_A_vis, self.desp_B_vis = self.remap_descriptor_pair(self.desp_A_vis, self.desp_B_vis)
        else:
            self.desp_A_vis = self.remap_descriptor_vis(self.desp_A_vis)
            self.desp_B_vis = self.remap_descriptor_vis(self.desp_B_vis)

    def remap_descriptor_pair(self, despA, despB):
        orig_shape = despA.shape
        desp_vec = np.vstack((despA.reshape(-1, orig_shape[2]), despB.reshape(-1, orig_shape[2])))
        desp_pca = self.pca_transform.transform(desp_vec)
        split = orig_shape[0] * orig_shape[1]
        despA, despB = desp_pca[0:split].reshape(orig_shape[0], orig_shape[1], 3), desp_pca[split:].reshape(
            orig_shape[0], orig_shape[1], 3)
        # set_trace()
        despA = self.remap_descriptor_vis(despA)
        despB = self.remap_descriptor_vis(despB)
        return despA, despB

    def remap_descriptor_vis_uniform(self, desp):
        for c in range(desp.shape[2]):
            c_max = desp[:, :, c].max()
            c_min = desp[:, :, c].min()
            scale = 1.0 / (c_max - c_min)
            desp[:, :, c] = ((desp[:, :, c] - c_min) * scale)
        # set_trace()
        return desp

    def remap_descriptor_vis(self, desp):  # desp is 480,640,D
        if desp.shape[2] > 3:  # Keep only 3 dimension
            desp = desp[:, :, 0:3]
        for c in range(desp.shape[2]):
            mean = desp[:, :, c].mean()
            std = desp[:, :, c].std()
            desp[:, :, c] = desp[:, :, c].clip(mean - std * self.conf.std_range / 2.0,
                                               mean + std * self.conf.std_range / 2.0)
            max_remap, min_remap = desp[:, :, c].max(), desp[:, :, c].min(),
            desp[:, :, c] = (desp[:, :, c] - min_remap) / (max_remap - min_remap)
        # set_trace()
        return desp

    def update_display(self, point_As=None, point_Bs=None):
        """
        :param point_As: list of tuples ((col,row),(B,G,R),radius)
        :param point_Bs: list of tuples ((col,row),(B,G,R),radius)
        :return:
        """
        # RGB to BGR, data type in np.float32, value range in [0,1]
        # TODO: Dimension reduction for higher descriptor dimension before calling this visualization function
        to_bgr = lambda x: np.ascontiguousarray(x[:, :, [2, 1, 0]])
        desp_A, desp_B, img_A, img_B, mask_A, mask_B = tuple(map(to_bgr,
                                                                 [self.desp_A_vis.copy(),
                                                                  self.desp_B_vis.copy(),
                                                                  self.rgb_A.copy(),
                                                                  self.rgb_B.copy(),
                                                                  self.mask_A.copy(),
                                                                  self.mask_B.copy()]))
        point_As = [] if point_As is None else point_As
        point_Bs = [] if point_Bs is None else point_Bs
        for center, color, radius in point_As:
            # desp_A = cv2.circle(desp_A, center, radius, color, thickness=-1)  # Filled circle
            img_A = cv2.circle(img_A, center, radius, color, thickness=-1)  # Filled circle
        for center, color, radius in point_Bs:
            # desp_B = cv2.circle(desp_B, center, radius, color, thickness=-1)  # Filled circle
            img_B = cv2.circle(img_B, center, radius, color, thickness=-1)  # Filled circle

        self.display_img = np.concatenate(
            (np.concatenate((img_A, desp_A), axis=0), np.concatenate((img_B, desp_B), axis=0),
             np.concatenate((mask_B, mask_A), axis=0)), axis=1)

    def set_threshold_bar(self, x):
        self.threshold = float(cv2.getTrackbarPos('threshold', 'bar')) / 10
        print('Threshold=', self.threshold)

    def set_query_point(self, event, col, row, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(row, col)
            if col < self.desp_A.shape[1]:
                # Query from A
                query_pt = [((col, row), colors['black'], 2)]
                desp = self.desp_A[row, col, :]
                closest_point, close_points = self.get_closest_and_threshold_descriptor_locs(desp, self.desp_B,
                                                                                             self.threshold)
                _, close_points_A = self.get_closest_and_threshold_descriptor_locs(desp, self.desp_A,
                                                                                   self.threshold)
                self.update_display(close_points_A + query_pt, close_points + [closest_point])
            else:
                # Query from B
                query_pt = [((col - self.desp_A.shape[1], row), colors['black'], 2)]
                desp = self.desp_B[row, col - self.desp_A.shape[1], :]
                closest_point, close_points = self.get_closest_and_threshold_descriptor_locs(desp, self.desp_A,
                                                                                             self.threshold)
                _, close_points_B = self.get_closest_and_threshold_descriptor_locs(desp, self.desp_B,
                                                                                   self.threshold)
                self.update_display(close_points + [closest_point], close_points_B + query_pt)

    @staticmethod
    def generate_region_idx(center, dist, image_size):
        h, w, _ = image_size
        r_ctr, c_ctr = center
        min_r, max_r = max(r_ctr - dist, 0), min(r_ctr + dist, h)
        min_c, max_c = max(c_ctr - dist, 0), min(c_ctr + dist, w)
        return min_r, max_r, min_c, max_c

    def set_query_vector(self, event, col, row, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(row, col)
            if col < self.desp_A.shape[1]:
                # Query from A
                query_pt = [((col, row), colors['black'], 2)]
                dim = self.desp_A.shape[2]
                min_r, max_r, min_c, max_c = self.generate_region_idx((row, col), 3, self.desp_A.shape)
                select_pt = self.desp_A[min_r:max_r, min_c:max_c].reshape(-1, dim)
                desp, std, mid = np.mean(select_pt, axis=0), np.std(select_pt, axis=0), np.median(select_pt,
                                                                                                  axis=0)
                print('Mean={}\n std={}\n mid={}'.format(desp, std, mid))

                closest_point, close_points = self.get_closest_and_threshold_descriptor_locs(desp, self.desp_B,
                                                                                             self.threshold)
                _, close_points_A = self.get_closest_and_threshold_descriptor_locs(desp, self.desp_A,
                                                                                   self.threshold)
                self.update_display(close_points_A + query_pt, close_points + [closest_point])
            else:
                # Query from B
                col_in_B = col - self.desp_A.shape[1]
                query_pt = [((col_in_B, row), colors['black'], 2)]
                dim = self.desp_B.shape[2]
                min_r, max_r, min_c, max_c = self.generate_region_idx((row, col_in_B), 3, self.desp_B.shape)
                select_pt = self.desp_B[min_r:max_r, min_c:max_c, :].reshape(-1, dim)
                desp, std, mid = np.mean(select_pt, axis=0), np.std(select_pt, axis=0), np.median(select_pt, axis=0)
                print('Mean={}\n std={}\n mid={}'.format(desp, std, mid))

                closest_point, close_points = self.get_closest_and_threshold_descriptor_locs(desp, self.desp_A,
                                                                                             self.threshold)
                _, close_points_B = self.get_closest_and_threshold_descriptor_locs(desp, self.desp_B,
                                                                                   self.threshold)
                self.update_display(close_points + [closest_point], close_points_B + query_pt)

    @staticmethod
    def get_closest_and_threshold_descriptor_locs(src_desp_value, tgt_desp_map, threshold=0.1):
        """
        :param src_desp_value: 1-D array of size N-Desc
        :param tgt_desp_map: [480,640,N-Desc]
        :param threshold: float, for descriptor
        :return: # ((col, row), (R, G, B), radius),[((col, row), (R, G, B), radius),......] which corresponds
                to "closest pt,within threshold pts....."
        """
        distance_map = np.linalg.norm(src_desp_value - tgt_desp_map, axis=2)
        # Closest Point
        closest_idx = np.argmin(distance_map)
        closest_pt = closest_idx % tgt_desp_map.shape[1], int(closest_idx / tgt_desp_map.shape[1])  # col,row

        close_pt_draw = (closest_pt, colors['red'], 3)

        # Calculate In-threshold Points
        close_rows, close_cols = np.where(distance_map < threshold)
        close_pts = [(p, colors['yellow'], 1) for p in tuple(zip(close_cols, close_rows))]

        return close_pt_draw, close_pts

    def draw_mask_from_label(self):
        mask_shape = self.desp_A.shape[0:2]
        mask_img_A = np.zeros((*mask_shape, 3))
        mask_img_B = np.zeros((*mask_shape, 3))
        for k, v in self.conf.mask_vis_colors.items():
            mask_img_A[self.label_A == k] = v
            mask_img_B[self.label_B == k] = v
        self.mask_A = mask_img_A
        self.mask_B = mask_img_B

    def get_pca_mapping(self):
        self.dataset.transform = lambda x: x  # Cancel normalization
        # rand_bk = self.dataset.randomize_background
        # self.dataset.randomize_background = False
        img_list = []
        for i in range(20):
            values = self.dataset[0]
            img_A, img_B = values[0], values[1]
            # 480,640,3  RGB format, np.float64 ,[0,1]
            self.rgb_A = img_A.astype(np.float32) / 255.0
            self.rgb_B = img_B.astype(np.float32) / 255.0

            # Forward Result one time
            print('Forwarding Images...')
            self.net.eval()
            height, width = self.rgb_A.shape[0], self.rgb_A.shape[1]
            assert self.model_conf.use_class
            with torch.no_grad():
                desp_A, desp_B = self.fwd(self.rgb_A.copy()), self.fwd(self.rgb_B.copy())
                label_A, label_B = torch.max(self.clf(desp_A.detach()), 1)[1], \
                                   torch.max(self.clf(desp_B.detach()), 1)[1]  # 307200,1
            label_A = label_A.cpu().numpy().reshape((height, width))
            label_B = label_B.cpu().numpy().reshape((height, width))
            desp_A = np.moveaxis(desp_A.cpu().numpy(), 0, 2)
            desp_B = np.moveaxis(desp_B.cpu().numpy(), 0, 2)
            img_list.extend([desp_A.reshape(-1, desp_A.shape[2]), desp_B.reshape(-1, desp_B.shape[2])])
        total_points = np.vstack(img_list)
        x, trans = run_PCA(total_points, k=3, return_transform=True)
        # set_trace()
        self.pca_transform = trans
        # self.dataset.randomize_background = rand_bk

    def run_visual_evaluation(self, args):
        cv2.namedWindow('bar')
        cv2.namedWindow('Evaluation')
        cv2.createTrackbar('threshold', 'bar', 0, 100, self.set_threshold_bar)
        if args.v2:
            cv2.setMouseCallback("Evaluation", self.set_query_vector)
        else:
            cv2.setMouseCallback("Evaluation", self.set_query_point)
        self.use_pca = args.pca
        if self.use_pca:
            self.get_pca_mapping()
        if args.i1 is None:
            self.sample_testing_images()
        else:
            base_dir = self.full_config['test_data']['base_dir']
            self.specify_testing_images(os.path.join(base_dir, args.i1), os.path.join(base_dir, args.i2))
        self.set_threshold_bar(None)
        self.update_display()
        while True:
            cv2.imshow("Evaluation", self.display_img)
            if cv2.waitKey(10) == ord('q'):
                break
            if cv2.waitKey(10) == ord('s'):
                self.sample_testing_images()
                self.update_display()

    """
    Functions for evaluation during training epochs
    """

    def run_training_evaluation(self, net,
                                loss_eval,
                                keep_desp=True,
                                pixd_use_class=False):  # Testing Data TODO: Separate loss_eval check dependency with trainer
        # Forward all values and record the result
        # loss_eval is an instance of MetricLoss
        self.device = loss_eval.device
        # Support multiple function call
        get_loss_maps = lambda func_name: getattr(loss_eval, func_name)().cpu()
        loss_maps = {'match': [],  # Init for append
                     'background': [],
                     'background_same': [],
                     'triplet_center': [],
                     'pixel_dist': []}
        if self.use_static:
            loss_maps['normalized_pixd'] = []
        if self.model_conf.use_class and pixd_use_class:
            loss_maps['pixel_dist_class'] = []
            if self.use_static:
                loss_maps['normalized_pixd_class'] = []

        desp_pairs = []  # For visualization
        img_pairs = []  # For visualization
        desp_remap = lambda x: tuple(
            self.remap_descriptor_vis(np.moveaxis(img[0].cpu().float().numpy(), 0, 2)) for img in x)
        img_denormalize = lambda x: tuple(denormalize_image(img[0].cpu().float()) for img in x)
        loss_recorder = ValueMeter()
        net.eval()
        device_img = self.model_conf.gpu_seq[0]
        with torch.no_grad():
            for i in range(len(self.test_data)):
                values = tuple(map(lambda x: x.clone(), self.test_data[i]))  # Copy

                if self.loss_conf.A_A2_only:
                    if self.model_conf.split_model:
                        img_A = values[0].to(device_img)
                        img_A2 = values[1].to(device_img)
                    else:
                        img_A = values[0].to(self.device)
                        img_A2 = values[1].to(self.device)

                    if self.model_conf.use_fp16:
                        img_A, img_A2 = tuple(map(lambda x: x.half(), [img_A, img_A2]))
                    desp_A = net(img_A).float()
                    desp_A2 = net(img_A2).float()
                    desp_B = values[2]
                    img_B = values[2]

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
                    desp_A = net(img_A).float()
                    desp_A2 = net(img_A2).float()
                    desp_B = net(img_B).float()
                if keep_desp:
                    img_pairs.append(img_denormalize([img_A, img_A2, img_B]))
                    desp_pairs.append(desp_remap([desp_A, desp_A2, desp_B]))  # Tuple of descriptor pair
                if self.use_static:
                    loss_eval.set_data(desp_A, desp_A2, desp_B, values[3:-1])  # One more for diagonal value
                else:
                    loss_eval.set_data(desp_A, desp_A2, desp_B, values[3:])
                loss_eval.set_centers(net.centers.float())
                # Get full loss record

                # get loss map for evaluation
                if self.loss_conf.split_match:
                    loss_obj_match, loss_obj_bk, loss_bk_bk = loss_eval.get_matching_loss(split=True)
                    loss_match = self.loss_conf.match_alpha_obj * loss_obj_match + \
                                 self.loss_conf.non_match_alpha_obj_bk * loss_obj_bk + \
                                 self.loss_conf.match_alpha_bk_bk * loss_bk_bk
                else:
                    loss_match = self.loss_conf.match_alpha * loss_eval.get_matching_loss()
                loss_hard = self.loss_conf.hard_alpha * loss_eval.get_hard_negative_loss()
                total_loss = loss_match + loss_hard
                if self.model_conf.use_class:
                    if self.loss_conf.use_hard_metric:
                        loss_class = self.loss_conf.clf_alpha * loss_eval.get_hard_metric_loss()
                    else:
                        loss_class = self.loss_conf.clf_alpha * loss_eval.get_discriminative_loss(net,
                                                                                                  self.model_conf.use_fp16)
                    total_loss += loss_class

                if not self.loss_conf.no_triplet:
                    loss_triplet = self.loss_conf.triplet_alpha * loss_eval.get_triplet_center_loss()
                    total_loss += loss_triplet

                if self.model_conf.learn_center:
                    centers_loss = self.loss_conf.center_alpha * loss_eval.get_loss_centers()
                    total_loss += centers_loss
                norm_loss = self.loss_conf.loss_norm_alpha * loss_eval.get_space_normalization_loss()
                total_loss += norm_loss

                pix_map = loss_eval.get_match_distance()  # for loss_recorder

                # loss_class = loss_eval.get_hard_metric_loss()  # for loss_recorder

                loss_recorder.record_data(loss_eval.get_loss_records())
                loss_recorder.record_data({'epoch_loss': total_loss.item()})
                loss_maps['match'].append(get_loss_maps('_same_objects_match'))
                loss_maps['background'].append(get_loss_maps('_objects_background'))
                loss_maps['background_same'].append(get_loss_maps('_background_same'))
                loss_maps['triplet_center'].append(get_loss_maps('_triplet_center_loss'))
                if self.model_conf.use_class and pixd_use_class:
                    closest_pixd, class_closest_pixd = \
                        getattr(loss_eval, '_match_closest_class_pixel_distance')(net, self.model_conf.use_fp16)
                    loss_maps['pixel_dist'].append(closest_pixd.cpu())  # Not sqrt ones
                    loss_maps['pixel_dist_class'].append(class_closest_pixd.cpu())
                else:
                    closest_pixd = get_loss_maps('_match_closest_pixel_distance')
                    loss_maps['pixel_dist'].append(closest_pixd)
                if self.use_static:
                    loss_maps['normalized_pixd'].append(closest_pixd / values[-1].item())
                    if self.model_conf.use_class and pixd_use_class:
                        loss_maps['normalized_pixd_class'].append(class_closest_pixd / values[-1].item())
            net.train()
            if self.multi_static:  # Multi_static
                del img_A, img_A2, img_B
                del net, loss_eval
                gc.collect()
                torch.cuda.empty_cache()
            for k in loss_maps.keys():  # Half,Sort and move to cpu
                loss_maps[k] = torch.sort(torch.cat(loss_maps[k]).half().to(self.device))[0].float().cpu().numpy()
            for k in ['match', 'background', 'background_same', 'triplet_center']:
                loss_maps[k] = np.sqrt(loss_maps[k])  # Undo pow(2), for correct understanding

        if self.use_static:
            if self.model_conf.use_class:
                pck_result = self.get_pck_evaluation(loss_maps['pixel_dist'])
                normalized_pck_result = self.get_normalized_pck_evaluation(loss_maps['normalized_pixd'])
                # Evaluation with class information
                if pixd_use_class:
                    pck_result.update(self.get_pck_evaluation(loss_maps['pixel_dist_class'], prefix='class-'))
                    normalized_pck_result.update(
                        self.get_normalized_pck_evaluation(loss_maps['normalized_pixd_class'], prefix='class-'))
                return loss_recorder.avg(), img_pairs, desp_pairs, loss_maps, pck_result, normalized_pck_result
            else:
                return loss_recorder.avg(), img_pairs, desp_pairs, loss_maps, self.get_pck_evaluation(
                    loss_maps['pixel_dist']), self.get_normalized_pck_evaluation(loss_maps['normalized_pixd'])
        else:
            if self.model_conf.use_class:
                pck_result = self.get_pck_evaluation(loss_maps['pixel_dist'])
                if pixd_use_class:
                    pck_result.update(self.get_pck_evaluation(loss_maps['pixel_dist_class'], prefix='class-'))
                return loss_recorder.avg(), img_pairs, desp_pairs, loss_maps, pck_result
            else:
                return loss_recorder.avg(), img_pairs, desp_pairs, loss_maps, self.get_pck_evaluation(
                    loss_maps['pixel_dist'])

    def get_pck_evaluation(self, pixel_dist_map, prefix=''):
        """
        :param pixel_dist_map: sorted pixel distance array
        :return:
        """
        ratio = np.linspace(0, 1, len(pixel_dist_map))
        result_dict = {}
        for pix in self.conf.pck_pix_list:
            r = np.where(pixel_dist_map > pix)[0]
            if len(r) == 0:
                val = np.array(1.0)
            else:
                val = ratio[r[0]]
            result_dict.update({prefix + 'PCK at pix=' + str(pix): val})
        result_dict.update({prefix + 'Mean_Pix-D_Exact': np.mean(pixel_dist_map)})
        return result_dict

    def get_normalized_pck_evaluation(self, normalized_pixel_dist_map, prefix=''):
        """
        :param normalized_pixel_dist_map: sorted normalized pixel distance array
        :return:
        """
        ratio = np.linspace(0, 1, len(normalized_pixel_dist_map))
        result_dict = {}
        for alpha in self.conf.pck_alpha_list:
            r = np.where(normalized_pixel_dist_map > alpha)[0]
            if len(r) == 0:
                val = np.array(1.0)
            else:
                val = ratio[r[0]]
            result_dict.update({prefix + 'PCK at alpha=' + str(alpha): val})
        result_dict.update({prefix + 'Mean_Pix-D_alpha_Exact': np.mean(normalized_pixel_dist_map)})
        return result_dict

    def plot_visualization(self, result_path, epoch, img_pairs, desp_pairs):
        pairs = []
        for pair_img, pair_desp in zip(img_pairs, desp_pairs):  # [[A,A2,B],[A_desp,A2_desp,AF_desp,B_desp]]
            pairs.append(np.concatenate((np.concatenate(pair_img, axis=1), np.concatenate(pair_desp, axis=1)), axis=0))
        pairs = pairs[0:self.conf.vis_pairs_n]
        pairs_stack = (np.concatenate(tuple(pairs), axis=0) * 255).astype(np.uint8)[:, :,
                      [2, 1, 0]]  # Vertical concatenate + BGR fix
        cv2.imwrite(os.path.join(result_path, 'visualization_epoch{}.png'.format(epoch)), pairs_stack)

    def plot_cdf(self, result_path, epoch, loss_maps):
        # fig.suptitle('Evaluation')
        # fig.subplots_adjust(wspace=0.64, hspace=0.48)
        fig, ax = plt.subplots(nrows=3, ncols=2)
        self.plot_cdf_subplot(ax[0, 0], 'match', self.loss_conf.match_m, 'Match L2', epoch, loss_maps['match'])
        self.plot_cdf_subplot(ax[0, 1], 'background', self.loss_conf.background_m,
                              'Background L2', epoch, loss_maps['background'])
        self.plot_cdf_subplot(ax[1, 0], 'triplet_center', self.loss_conf.center_dist_diff,
                              'Triplet Center', epoch, loss_maps['triplet_center'])
        self.plot_cdf_subplot(ax[1, 1], 'background_same', self.loss_conf.background_consistent_m,
                              'Background force same', epoch, loss_maps['background_same'])
        self.plot_cdf_subplot(ax[2, 0], 'pixel_dist', self.conf.pixel_vis_scale,
                              'Match Pixel Distance', epoch, loss_maps['pixel_dist'])

        fig.suptitle('Evaluation')
        # plot_cdf_subplot((3, 3, 6),,, )
        # plot_cdf_subplot((3, 3, 7),,, )
        # plot_cdf_subplot((3, 3, 8),,, )
        # plot_cdf_subplot((3, 3, 9),,, )
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(result_path, 'evaluation_epoch{}.png'.format(epoch)), dpi=300)
        try:
            plt.close('all')
        except:
            pass

    def plot_cdf_subplot(self, ax, key, loss_m, title, epoch, loss_map):
        x_max = self.conf.x_lim_multi * loss_m
        y_accu = np.linspace(0, 1, len(loss_map))
        ax.set_xlabel(title)
        ax.set_ylabel('Fraction of image')
        ax.grid(True)
        ax.set_ylim(0, 1.01)
        ax.set_xlim(0, x_max)
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.set_tick_params(labelsize=4)
        ax.yaxis.set_tick_params(labelsize=4)
        ax.set_xticks(np.linspace(0, x_max, self.conf.x_lim_multi + 1))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.plot(loss_map, y_accu, 'k-',
                linewidth=2, label='epoch: %d' % epoch)
        ax.legend(loc='lower right', fontsize='xx-small')

    def load_config(self, args):
        # Load yaml configs
        # Command line arguments override the yaml configs if provided
        with open(args.config, 'r') as stream:
            self.full_config = yaml.load(stream)  # Full config for reference
        if args.overwrite_test is not None:
            with open(args.overwrite_test, 'r') as stream:
                overwrite_test_config = yaml.load(stream)  # Full config for reference
            self.full_config['sampling'] = overwrite_test_config['sampling']
            self.full_config['test_data'] = overwrite_test_config['test_data']
            self.full_config['evaluation'] = overwrite_test_config['evaluation']
            print('Descriptor Evaluation: Overwrite sampling/test data/evaluation config ')
        # the keys of "full_config" is the first layer key in the config yaml file
        self.conf = ConfNamespace(self.full_config['evaluation'], args.__dict__)
        print('Test set n=', self.conf.test_set_n)
        self.model_conf = ConfNamespace(self.full_config['model'])
        self.tr_conf = ConfNamespace(self.full_config['training_param'])
        self.loss_conf = ConfNamespace(self.full_config['criterion'])

    def generate_PCA_scene(self, ):
        self.get_pca_mapping()
        self.use_pca = True
        to_bgr = lambda x: np.ascontiguousarray(x[:, :, [2, 1, 0]])
        dir_name, N = self.full_config['visual_data']['id_to_scenes'][self.full_config['visual_data']['use_id']][0]
        base_dir = os.path.join(self.full_config['test_data']['base_dir'], dir_name)
        output_base = self.full_config['visual_data']['output_base']
        from utils.training_utils import check_path
        write_dir = os.path.join(output_base, dir_name)
        check_path(write_dir)

        print('Dir=', dir_name, ' N=', N)
        for i in range(0, N, 2):
            print('Processing ', i, ' ', i + 1)
            self.specify_testing_images(os.path.join(base_dir, '{:06}-color.png'.format(i)),
                                        os.path.join(base_dir, '{:06}-color.png'.format(i + 1)))
            desp_A, desp_B = tuple(map(to_bgr, [self.desp_A_vis.copy(), self.desp_B_vis.copy()]))
            desp_A = (255 * desp_A).astype(np.uint8)  # Now scale by 255
            desp_B = (255 * desp_B).astype(np.uint8)
            # img = data.astype(np.uint8)
            cv2.imwrite(os.path.join(write_dir, '{:06}-color.png'.format(i)), desp_A)
            cv2.imwrite(os.path.join(write_dir, '{:06}-color.png'.format(i + 1)), desp_B)


def denormalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert img.size(0) == 3, 'Image is in C,H,W format,float tensor'
    inv_normalize = tf.Normalize(
        mean=np.divide(-np.array(mean), np.array(std)),
        std=1 / np.array(std)
    )
    img = inv_normalize(img)
    img = np.moveaxis(img.numpy(), 0, 2)  # 480,640,3 float
    return img


def run_PCA(data, k=3, batch=300000, return_transform=False):
    # transformer = IncrementalPCA(n_components=k, batch_size=batch)
    transformer = PCA(n_components=k)
    print('Loading Data in PCA')
    X_transformed = transformer.fit_transform(data)
    print('PCA Done')
    if return_transform:
        return X_transformed, transformer
    else:
        return X_transformed, None


def simple_path_sampler(data_config, sample_config):
    objA_class = np.random.choice(data_config.object_classes, p=np.asarray(data_config.class_match_sample_weight) / sum(
        data_config.class_match_sample_weight))
    objA_id = np.random.choice(data_config.class_to_object_ids[objA_class])
    sceneA_sub_idx = np.random.choice(len(data_config.id_to_scenes[objA_id]))
    folderA, imgA_N = data_config.id_to_scenes[objA_id][sceneA_sub_idx]

    while True:
        objB_class = np.random.choice(data_config.object_classes,
                                      p=np.asarray(data_config.class_match_sample_weight) / np.sum(
                                          data_config.class_match_sample_weight))
        if objB_class != objA_class:
            break
    objB_id = np.random.choice(data_config.class_to_object_ids[objB_class])
    sceneB_sub_idx = np.random.choice(len(data_config.id_to_scenes[objB_id]))
    folderB, imgB_N = data_config.id_to_scenes[objB_id][sceneB_sub_idx]

    img_idx_A = np.random.choice(imgA_N)
    img_idx_A2 = img_idx_A
    while img_idx_A2 == img_idx_A:
        img_idx_A2 = np.random.randint(img_idx_A - sample_config.match_frame_near * 3,
                                       img_idx_A + sample_config.match_frame_near * 3 + 1) % imgA_N
    img_idx_AF = (img_idx_A + int(imgA_N / 2)) % imgA_N
    img_idx_B = np.random.choice(imgB_N)

    # Open the images
    path_img = lambda folder, img_type, idx: os.path.join(data_config.base_dir, folder,
                                                          '{:06}-{}.png'.format(idx, img_type))
    path_A, path_A2, path_AF, path_B = tuple(
        map(path_img, [folderA, folderA, folderA, folderB], ['color', 'color', 'color', 'color'],
            [img_idx_A, img_idx_A2, img_idx_AF, img_idx_B]))
    return path_A, path_A2


def main():
    os.putenv('DISPLAY', ':0')
    """
    Press q to exit
    Press s to sample new pairs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Training Config file is required", type=str)
    parser.add_argument('-tps', help="Training Progress number for restore", type=int)
    parser.add_argument('-overwrite_test', help='Overwrite Test Setting', type=str, default=None)
    parser.add_argument('--eval', help='Evaluation on visual_data folders', action='store_true', default=False)
    parser.add_argument('--pca', help='Visuaslization using PCA', action='store_true', default=False)
    parser.add_argument('--scene', help='Generate Scene PCA projection', action='store_true', default=False)
    parser.add_argument('--v2', help='EvaluationV2', action='store_true', default=False)
    parser.add_argument('-i1', help='Image 1 Path', type=str, default=None)
    parser.add_argument('-i2', help='Image 2 Path', type=str, default=None)
    args = parser.parse_args()
    # net_path = './progress/group1/network.pth'
    # img1_path = './Data_Real/0008/000000-color.png'
    # img2_path = './Data_Real/0008/000002-color.png'
    # loader = DescriptorEvaluation(net_path, img1_path, img2_path)
    loader = DescriptorEvaluation(args=args)
    if args.scene:
        loader.generate_PCA_scene()
    else:
        loader.run_visual_evaluation(args)
    # else:
    #     loader.run_instance_evaluation()


if __name__ == '__main__':
    main()

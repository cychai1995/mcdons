import numpy as np
import os, sys

from PIL import Image  # CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
from torchvision import transforms
from torch.utils.data import Dataset
import cv2  # For testing code only
from sklearn.neighbors import NearestNeighbors

#
# from ipdb import set_trace
"""
StaticDataset: For stable testing result, fixed random seed
A,A2 Only , No patch and multi patch
"""


class StaticDataset(Dataset):
    def __init__(self, sampling_config, data_config):
        # Access all parameters directly
        self.__dict__.update(sampling_config)
        self.__dict__.update(data_config)
        transform_list = []
        transform_list.extend([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
        self.transform = transforms.Compose(transform_list)
        self.cam_mat_inv = np.linalg.inv(self.cam_mat)
        # TODO: use self.object_classes for the full remap dict, but all sampling only A,A2 (match class only)
        self.clf_remap_dict = {0: 0}
        self.clf_remap_dict.update({k: i + 1 for i, k in enumerate(self.object_classes)})
        # Match class and non-match classes are fixed in StaticDataset, check all folder here
        assert len(self.match_classes) == 1, 'Static Dataset only run in single object scenes'
        self.match_class = self.match_classes[0]  # Single match class
        match_ids = self.class_to_object_ids[self.match_class]  # Class to object idss
        assert len(match_ids) == 1, 'Static Dataset only run in single class instance'
        self.match_id = match_ids[0]
        # Calculate the total len
        self.folders = self.id_to_scenes[self.match_id]

        self.sample_len = 0
        folder_cnt = []
        for fd_name, n_view in self.folders:
            self.sample_len += n_view - self.match_frame_near  #
            folder_cnt.append(self.sample_len)
        self.folder_cnt = np.asarray(folder_cnt)
        self.rs = np.random.RandomState(1234)

    def __getitem__(self, idx):  # Sampling Dataset
        # Sequentially
        """
        :param idx: Useless
        :return:
        Matching Class Set: K classes
        Non-Matching Class set: M classes
        img_A: containing Na objects from K
        img_A2: containing Na objects from K
        img_B: containing Nb objects from M

        Unique indices for each pixel points

        Sampling Strategy
        Sample K,M classes (Mutually Exclusive) -> Sample one scene for each class
        -> Sample 2 images from each scene for img_A and img_A2 (total n(K) scenes)
        -> Sample 1 image from each scene for img_B (total n(M) scenes)

        # All indices is row-majored reshaped image indices

        #>>> Part1 Total Data Points <<<
        mask_A_1d: indices of img_A that contain object A
        bg_A_1d: background indices in img_A (not contain any other objects)

        mask_A2_1d: all indices of img_A2 that contain object A
        bg_A2_1d: background indices in img_A2 (not contain any other objects)

        mask_B_1d: all indices of img_B that contain object B
        bg_B_1d: background indices in img_B (not contain any other objects)

        match_idx_A:
        match_idx_A2: All img_A,img_A2 matched indices (pair-order indices)
        # Note: Shuffle the paired indices to sample same object but non-matched pairs

        no_match_idx_A
        no_match_idx_A2: All no-match found indices (non-ordered indices, but not the background)

        # Note: Union(non_match_idx_A,match_idx_A) = mask_idx_A
        """
        # Sample scene A and object A(A2) id
        np.seterr(divide='ignore', invalid='ignore')

        while True:
            chk_match = 0
            while chk_match < 1:
                """
                match/non-match class set -> match/non-match object ids -> sample folders for each object
                -> sample match paired images ->extract foreground patches
                -> merge patches and match indices
                """
                class_mapping_dict = {self.match_id: self.match_class}
                folder_idx = np.where(self.folder_cnt > idx)[0][0]
                folder_name = self.folders[folder_idx][0]  # ge the name from [[Name,N]]
                if folder_idx > 0:
                    p_start = idx - self.folder_cnt[folder_idx - 1]
                else:
                    p_start = idx
                p_end = p_start + self.rs.randint(1, self.match_frame_near + 1)  # TODO: Fixed seed randomization here
                # [img_m11,img_m12,img_m21,img_m22,...], 2D image
                match_ids = [self.match_id]
                match_folder_ids = [(folder_name, [p_start, p_end])]
                match_img = [self.open_img(folder, 'color', p, 0)
                             for folder, pair in match_folder_ids for p in pair]
                img_A, img_A2 = match_img
                # [[mask2d11,mask2d12],[mask2d21,mask2d22],...], 2d mask
                match_mask = [[self.open_img(folder, 'label', pair[0], 1),
                               self.open_img(folder, 'label', pair[1], 1)] \
                              for folder, pair in match_folder_ids]

                # [mask11,mask12,mask21,mask22,...], 1d mask
                match_mask_1d = [np.where(mask2d.reshape(-1) == obj_id)[0] for masks, obj_id in
                                 zip(match_mask, [self.match_id]) for mask2d in masks]
                match_bg_1d = [np.where(mask2d.reshape(-1) == 0)[0] for masks in match_mask for mask2d in masks]
                match_diag = get_box_diagonal_length(img_A2.shape, match_mask_1d[1])
                # [depth11,depth12,depth21,depth22,...], 2D depth
                match_depth = [self.open_img(folder, 'depth', p, 1)
                               for folder, pair in match_folder_ids for p in pair]
                match_transforms = [self.relative_transform(folder, pair[0], pair[1]) for folder, pair in
                                    match_folder_ids]

                # (src,tgt) for each matching pair

                match_src_indices = []  # For pair src
                match_tgt_indices = []  # For pair tgt
                for i in range(0, len(match_mask_1d), 2):  # 2 images per pair match
                    src_indices, tgt_indices = self.projection_match(match_depth[i], match_mask_1d[i], \
                                                                     match_depth[i + 1], match_mask_1d[i + 1], \
                                                                     match_transforms[i // 2],
                                                                     self.dual_projection_check)
                    match_src_indices.append(src_indices)
                    match_tgt_indices.append(tgt_indices)

                chk_match = np.min([s.shape[0] for s in match_src_indices])
                if chk_match == 0:
                    print('One of the match is 0 after projection, resample')
                    continue
                # Keep only patch information (A little bit of bk, mostly object)
                # for i in range(len(match_img)):
                #     cv2.imshow('Test', match_img[i])
                #     while True:
                #         if cv2.waitKey(10) == ord('s'):
                #             break

                # BGR -> RGB for images (CV2)

            # Match result for sampling
            # All id is object id
            match_idx_A = np.hstack(match_src_indices)
            match_idx_A2 = np.hstack(match_tgt_indices)
            match_class_id = np.repeat(match_ids, [len(m) for m in match_src_indices])
            mask_A_1d = match_mask_1d[0]
            mask_A2_1d = match_mask_1d[1]

            fg_A_cls, fg_A2_cls = np.repeat(self.match_class, len(mask_A_1d)), np.repeat(self.match_class,
                                                                                         len(mask_A2_1d))

            bg_A_1d = match_bg_1d[0]
            bg_A2_1d = match_bg_1d[1]

            """
            >>> Part2 Sample Data Points <<<
            Positive, Negative....
            """
            # For match -> As much as possible, not balance
            if self.balanced_match:
                s_match_idx_A, s_match_idx_A2 = self.sample_balanced_pairs(match_class_id, match_idx_A, match_idx_A2,
                                                                           self.match_n)
                s_match_hn_A, s_match_hn_A2 = self.sample_balanced_pairs(match_class_id, match_idx_A, match_idx_A2,
                                                                         self.hard_negative_match)
            else:
                s_match_idx_A, s_match_idx_A2 = self.sample_match_pairs(match_idx_A, match_idx_A2, N=self.match_n)
                s_match_hn_A, s_match_hn_A2 = self.sample_match_pairs(match_idx_A, match_idx_A2,
                                                                      N=self.hard_negative_match)
            # Non_match(_contrastive_closest_non_matchAA2)
            # non_match_idx_A, non_match_idx_A2 = self.calculate_non_match_expand(match_idx_A, match_idx_A2, mask_A_1d,
            #                                                                     mask_A2_1d, img_A.shape[0:2],
            #                                                                     self.non_match_kernel)

            # For Foreground <-> Background M (_objects_background)
            s_fg_A = self.sample_balanced_elements(fg_A_cls, mask_A_1d, self.background_n_per_class)
            s_bg_A = self.sample_elements_1d(bg_A_1d, len(s_fg_A))

            s_fg_A2 = self.sample_balanced_elements(fg_A2_cls, mask_A2_1d, self.background_n_per_class)
            s_bg_A2 = self.sample_elements_1d(bg_A2_1d, len(s_fg_A2))

            # Background Same (_background_same)
            s_bg_A_same = self.sample_elements_1d(bg_A_1d, self.background_same_n)
            s_bg_A2_same = self.sample_elements_1d(bg_A2_1d, self.background_same_n)
            # s_bg_B_same = self.sample_elements_1d(bg_B_1d, self.background_same_n)

            # For closest pixel evaluation(_match_closest_pixel_distance)
            s_match_pix_A, s_match_pix_A2 = self.sample_balanced_pairs(match_class_id, match_idx_A, match_idx_A2,
                                                                       self.pix_dist_match_n)
            # For classification(mask)
            s_cls_label_A, s_cls_A = self.sample_balanced_elements(fg_A_cls, mask_A_1d, self.segmentation_n // 2,
                                                                   return_label=True)
            s_cls_label_A2, s_cls_A2 = self.sample_balanced_elements(fg_A2_cls, mask_A2_1d, self.segmentation_n // 2,
                                                                     return_label=True)
            s_cls_bk_A = self.sample_elements_1d(bg_A_1d, self.segmentation_n)

            # s_cls_labels = np.hstack((s_cls_label_A, s_cls_label_A2, np.zeros_like(s_cls_bk_A), s_cls_label_B))
            # Mapping class label from class_id to 0~N_classes that actually used in the sampling config, for segmentation prediction
            # if self.A_A2_only:
            s_cls_labels = np.hstack((s_cls_label_A, s_cls_label_A2, np.zeros_like(s_cls_bk_A)))
            s_cls_labels = self.remap_label_lists([s_cls_labels], self.clf_remap_dict)[0]

            # Sample mask for cross non_match (_contrastive_closest_non_matchAB)
            hn_sample_n = np.min([len(mask_A_1d), len(mask_A2_1d), self.hard_negative_AB_n])
            s_mask_A_hn = self.sample_elements_1d(mask_A_1d, hn_sample_n)
            s_mask_A2_hn = self.sample_elements_1d(mask_A2_1d, hn_sample_n)
            # Sample mask for _contrastive_closest_non_matchAA2, too big mask will use too many memory
            s_mask_A_hn_AA2 = self.sample_elements_1d(mask_A_1d, self.mask_max_hard_negative)
            s_mask_A2_hn_AA2 = self.sample_elements_1d(mask_A2_1d, self.mask_max_hard_negative)

            # BGR -> RGB
            img_A = img_A[:, :, [2, 1, 0]]
            img_A2 = img_A2[:, :, [2, 1, 0]]
            # img_B = img_B[:, :, [2, 1, 0]]
            # Transform
            img_A, img_A2 = tuple(map(self.transform, [img_A, img_A2]))
            match_class_set = np.array([self.match_class])
            empty_lst = np.array([-1])
            return_vals = (img_A, img_A2, np.zeros_like(img_A),
                           mask_A_1d, mask_A2_1d, empty_lst, \
                           fg_A_cls, fg_A2_cls, empty_lst, \
                           bg_A_1d, bg_A2_1d, empty_lst, \
                           match_class_set, empty_lst, \
                           s_match_idx_A, s_match_idx_A2, s_match_hn_A, s_match_hn_A2, \
                           s_mask_A_hn_AA2, s_mask_A2_hn_AA2, \
                           s_fg_A, s_bg_A, s_fg_A2, s_bg_A2, empty_lst, empty_lst, \
                           s_bg_A_same, s_bg_A2_same, empty_lst, \
                           s_match_pix_A, s_match_pix_A2, \
                           s_mask_A_hn, s_mask_A2_hn, \
                           empty_lst,empty_lst,empty_lst,\
                           s_cls_A, s_cls_A2, empty_lst, s_cls_bk_A, s_cls_labels, match_diag)
            if self.return_checker(return_vals):
                break
            else:
                print('Dataset Re-sample, Return Checker Fails')
        return return_vals

    def remap_label_lists(self, id_list, mapping):
        """
        :param id_list: list of 1d ndarray
        :param mapping: dictionary of m
        :return:
        """
        return [replace_with_dict(l, mapping) for l in id_list]

    def sample_balanced_elements(self, class_label, elements, N_per_class, return_label=False):
        """
        :param class_label: integer class label in ndarray
        :param elements: list of items for sampling
        :param N: N for each class for sampling
        :return: s_class_label,s_elements
        """
        labels, counts = np.unique(class_label, return_counts=True)
        sample_n = min(np.min(counts), N_per_class)
        s_elements = np.hstack([self.rs.choice(elements[class_label == l], sample_n) for l in labels])
        if return_label:
            return np.repeat(labels, sample_n), s_elements
        else:
            return s_elements

    def sample_balanced_pairs(self, class_label, elements_1, elements_2, max_N, return_label=False):
        """
        :param class_label: integer class label in ndarray ,for pair elements_1,elements_2
        :param elements_1: list of items for sampling
        :param elements_2: list of items for sampling
        :param N: N for each class for sampling
        :return: s_class_label,s_elements
        """
        labels, counts = np.unique(class_label, return_counts=True)
        sample_n = min(np.min(counts), max_N // len(labels))
        s_1 = []
        s_2 = []
        for l in labels:
            select = self.rs.choice(np.where(class_label == l)[0], sample_n)
            s_1.append(elements_1[select])
            s_2.append(elements_2[select])
        if return_label:
            return np.repeat(labels, sample_n), np.hstack(s_1), np.hstack(s_2)
        else:
            return np.hstack(s_1), np.hstack(s_2)

    def sample_elements_1d(self, elements, N):
        N_sample = min(len(elements), N)
        idx = random_permute_derangement(len(elements), self.rs)
        return elements[idx[0:N_sample]]

    def projection_match(self, src_depth_img, src_obj_mask, tgt_depth_img, tgt_obj_mask, tgtTsrc, dual_check=False):
        """
        :param src_depth_img: 2D depth image
        :param src_obj_mask: 1D pixel indices that contains the specific object in src image
        :param tgt_depth_img: 2D depth image
        :param tgt_obj_mask: 1D pixel indices that contains the specific object in target image
        :param tgtTsrc: tgtTsrc: 4*4 transform matrix tgt<-src
        :param dual_check: Dual Projection check for matching
        :return: two 1D pixel index array for src frame and target frame, paired order
        """
        width, height = src_depth_img.shape[1], src_depth_img.shape[0]
        u_src = np.tile(np.arange(0, width), height)
        v_src = np.repeat(np.arange(0, height), width)
        z_src = src_depth_img.reshape(-1) * self.depth_scale

        uz_src, yz_src = np.multiply(u_src, z_src), np.multiply(v_src, z_src)
        # TODO: Apply src object mask, then calculate source x,y,z (may slightly faster?)
        uvz_src = np.vstack((uz_src, yz_src, z_src))  # [src_obj_mask]
        xyz_src = np.dot(self.cam_mat_inv, uvz_src)  # 3,len(src_obj_mask)
        xyz_src_homo = np.vstack((xyz_src, np.ones(xyz_src.shape[1])))
        # Project to target frame
        xyz_tgt = np.dot(tgtTsrc, xyz_src_homo)[0:3, :]
        uvz_tgt = np.dot(self.cam_mat, xyz_tgt)
        u_tgt = np.clip(np.round(np.divide(uvz_tgt[0], uvz_tgt[2])).astype(np.int32), a_min=0, a_max=width - 1)
        v_tgt = np.clip(np.round(np.divide(uvz_tgt[1], uvz_tgt[2])).astype(np.int32), a_min=0, a_max=height - 1)
        # For nan case
        z_tgt_projected = uvz_tgt[2]  # From source frame and pose
        z_tgt_depth = tgt_depth_img[v_tgt, u_tgt]  # From depth measurement, query projected points, # 640*480 values

        # Compared calculated z and true depth value
        match_depth_indices = \
            np.where(np.abs(z_tgt_depth * self.depth_scale - z_tgt_projected) < self.point_distance_check)[0]

        # Construct match indices in 1-D

        # pixel indices for target image
        match_idx_tgt = v_tgt[match_depth_indices] * width + u_tgt[match_depth_indices]
        # pixel indices for source image
        match_idx_src = match_depth_indices

        # Filter-out background and other object
        match_idx_tgt, keep_indices_tgt, _ = np.intersect1d(match_idx_tgt, tgt_obj_mask, return_indices=True)
        match_idx_src, keep_indices_src, _ = np.intersect1d(match_idx_src[keep_indices_tgt], src_obj_mask,
                                                            return_indices=True)
        match_idx_tgt = match_idx_tgt[keep_indices_src]  # Foreground object in both sides
        assert match_idx_tgt.shape[0] == match_idx_src.shape[0]
        # match_idx_src, match_idx_tgt

        # print('Match Points', match_idx_tgt.shape[0])
        if dual_check:
            # match_idx_A_AA2, match_idx_A2_AA2 = self.projection_match(depth_A, mask_A_1d, depth_A2, mask_A2_1d,
            #                                                           A2_T_A)
            match_idx_tgt2, match_idx_src2 = self.projection_match(tgt_depth_img, tgt_obj_mask, src_depth_img,
                                                                   src_obj_mask, np.linalg.inv(tgtTsrc),
                                                                   dual_check=False)
            match_idx_src, keep_src_idx, _ = np.intersect1d(match_idx_src, match_idx_src2, return_indices=True)
            match_idx_tgt, keep_tgt_idx, _ = np.intersect1d(match_idx_tgt[keep_src_idx], match_idx_tgt2,
                                                            return_indices=True)
            match_idx_src = match_idx_src[keep_tgt_idx]

        return match_idx_src, match_idx_tgt  # 1D index in image

    def sample_match_pairs(self, match_idx_A, match_idx_A2, N):  # Match
        assert len(match_idx_A) == len(match_idx_A2)
        N_sample = min(len(match_idx_A), N)
        rand_idx = self.rs.permutation(len(match_idx_A))[0:N_sample]
        return match_idx_A[rand_idx], match_idx_A2[rand_idx]

    def open_img(self, folder, info_type, idx, mode):
        if mode == 0:  # RGB
            return cv2.imread(os.path.join(self.base_dir, folder, '{:06}-{}.png'.format(idx, info_type)))
        else:  # Mask,Depth
            return cv2.imread(os.path.join(self.base_dir, folder, '{:06}-{}.png'.format(idx, info_type)),
                              cv2.IMREAD_ANYDEPTH)

    def relative_transform(self, folder, idx1, idx2):
        # Retuen the transformation of idx2_T_idx1
        pose_1 = np.loadtxt(
            os.path.join(self.base_dir, folder, '{:06}-pose.txt'.format(idx1)))
        pose_2 = np.loadtxt(
            os.path.join(self.base_dir, folder, '{:06}-pose.txt'.format(idx2)))
        return np.dot(np.linalg.inv(pose_2), pose_1)

    def __len__(self):
        return self.sample_len

    @staticmethod
    def return_checker(return_tuple):
        for item in return_tuple:
            if isinstance(item, np.ndarray):
                if item.shape[0] == 0:
                    return False
            if isinstance(item, list):
                for k in item:
                    if isinstance(item, np.ndarray):
                        if item.shape[0] == 0:
                            return False
        return True


def cv2_show_img_lists(img_list, msg=''):
    print(msg)
    print('s for next, q for exit')
    for i in range(len(img_list)):
        cv2.imshow('Test', img_list[i])
        while True:
            if cv2.waitKey(10) == ord('s'):
                break


# https://stackoverflow.com/questions/47171356/replace-values-in-numpy-array-based-on-dictionary-and-avoid-overlap-between-new
def replace_with_dict(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    return vs[np.searchsorted(ks, ar)]


# Guarantee no element stays in the same position after random permutation, for indexing only
def random_permute_derangement(N, np_rs):
    original = np.arange(N)
    new = np_rs.permutation(N)
    same = np.where(original == new)[0]
    if N == 1:
        # Warning("Invalid Derangement")
        return [0]
    while len(same) != 0:
        swap = same[np_rs.permutation(len(same))]
        new[same] = new[swap]
        same = np.where(original == new)[0]
        if len(same) == 1:
            swap = np_rs.randint(0, N)
            new[[same[0], swap]] = new[[swap, same[0]]]
    return new


def get_box_diagonal_length(img_shape, mask_1d):
    h, w, _ = img_shape
    u, v = mask_1d % w, mask_1d // w
    u1, u2 = np.min(u), np.max(u)
    v1, v2 = np.min(v), np.max(v)
    diag = np.sqrt((u1 - u2) ** 2 + (v1 - v2) ** 2)
    return diag


def static_dataset_vis_test():
    disp_max = 5000
    os.putenv('DISPLAY', ':0')
    import yaml
    with open('/home/chai/projects/visual_descriptor_learning/exps/temporary/0113_test_static.yaml',
              'r') as f:
        dd = yaml.load(f)
    pd = StaticDataset(dd['sampling'], dd['test_data'])
    pd.transform = lambda x: x  # No normalization, for visualization
    pd.match_n = disp_max
    pd.background_n = 50000
    keys = ['mask_A_1d', 'mask_A2_1d', 'mask_B_1d', \
            'fg_A_cls', 'fg_A2_cls', 'fg_B_cls', \
            'match_class_set', 'non_match_class_set', \
            's_match_idx_A', 's_match_idx_A2', 's_match_hn_A', 's_match_hn_A2', \
            's_mask_A_hn_AA2', 's_mask_A2_hn_AA2',
            's_fg_A', 's_bg_A', 's_fg_A2', 's_bg_A2', 's_fg_B', 's_bg_B', \
            's_bg_A_same', 's_bg_A2_same', 's_bg_B_same', \
            's_match_pix_A', 's_match_pix_A2', \
            's_mask_A_hn', 's_mask_A2_hn', \
            's_cls_A', 's_cls_A2', 's_cls_B', 's_cls_bk_A', 's_cls_labels']
    test_pairs = [[0, 1, False], [8, 9, True], [10, 11, True], [12, 13, False],
                  [14, 16, False], [15, 17, False],
                  [20, 21, False], [23, 24, True],
                  [25, 26, False], [27, 28, False]]

    while True:
        values = pd[0]
        # set_trace()
        img_A, img_A2, img_B, match_idx_src, match_idx_tgt = \
            values[0], values[1], values[2], values[11], values[12]  # A AF
        data = values[3:]

        print('Match N=', match_idx_src.shape[0])
        img_concate3 = np.concatenate((img_A, img_A2), axis=1)
        img_concate3 = np.ascontiguousarray(img_concate3[:, :, [2, 1, 0]])
        img_concate2 = np.concatenate((img_A, img_A2), axis=1)
        img_concate2 = np.ascontiguousarray(img_concate2[:, :, [2, 1, 0]])
        cv2.imshow("Test", img_concate3)
        while True:
            if cv2.waitKey(10) == ord('s'):
                break
        for ti, tp in enumerate(test_pairs):
            # if ti == 2:
            #     break
            if len(tp) == 3:  # A A2
                disp_img = np.array(img_concate2)
                src_u, src_v = indices1d_remap_new_uv(data[tp[0]], img_A.shape, (480, 640, 3))
                tgt_u, tgt_v = indices1d_remap_new_uv(data[tp[1]], img_A2.shape, (480, 640, 3), shift=(640, 0))
                sample_n = min(len(src_u), len(tgt_u))
                step = max(1, int(sample_n // disp_max))
                if tp[2]:
                    for i in range(0, sample_n, step):
                        pts_src = src_u[i], src_v[i]
                        pts_tgt = tgt_u[i], tgt_v[i]
                        colors = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                        colors = (int(colors[0]), int(colors[1]), int(colors[2]))
                        cv2.circle(disp_img, pts_src, 2, colors, thickness=-1)  # Filled circle
                        cv2.circle(disp_img, pts_tgt, 2, colors, thickness=-1)  # Filled circle
                        cv2.line(disp_img, pts_src, pts_tgt, colors)
                else:
                    for i in range(0, len(src_u), step):
                        pts_src = src_u[i], src_v[i]
                        colors = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                        colors = (int(colors[0]), int(colors[1]), int(colors[2]))
                        cv2.circle(disp_img, pts_src, 2, colors, thickness=-1)  # Filled circle
                    for i in range(0, len(tgt_u), step):
                        pts_tgt = tgt_u[i], tgt_v[i]
                        colors = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                        colors = (int(colors[0]), int(colors[1]), int(colors[2]))
                        cv2.circle(disp_img, pts_tgt, 2, colors, thickness=-1)  # Filled circle

                print(keys[tp[0]], ' & ', keys[tp[1]])
            else:  # A,A2,B
                disp_img = np.array(img_concate3)
                src_u, src_v = indices1d_remap_new_uv(data[tp[0]], img_A.shape, (480, 640, 3))
                tgt_u, tgt_v = indices1d_remap_new_uv(data[tp[1]], img_A2.shape, (480, 640, 3), shift=(640, 0))
                bu, bv = indices1d_remap_new_uv(data[tp[2]], img_B.shape, (480, 640, 3), shift=(1280, 0))
                sample_n = np.min([len(src_u), len(tgt_u), len(bu)])
                step = max(1, int(sample_n // disp_max))
                for i in range(0, len(src_u), step):
                    pts_src = src_u[i], src_v[i]
                    colors = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                    colors = (int(colors[0]), int(colors[1]), int(colors[2]))
                    cv2.circle(disp_img, pts_src, 2, colors, thickness=-1)  # Filled circle
                for i in range(0, len(tgt_u), step):
                    pts_src = tgt_u[i], tgt_v[i]
                    colors = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                    colors = (int(colors[0]), int(colors[1]), int(colors[2]))
                    cv2.circle(disp_img, pts_src, 2, colors, thickness=-1)  # Filled circle
                for i in range(0, len(bu), step):
                    pts_src = bu[i], bv[i]
                    colors = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                    colors = (int(colors[0]), int(colors[1]), int(colors[2]))
                    cv2.circle(disp_img, pts_src, 2, colors, thickness=-1)  # Filled circle

                print(keys[tp[0]], ' & ', keys[tp[1]], ' & ', keys[tp[2]])
            cv2.imshow("Test", disp_img)
            while True:
                if cv2.waitKey(10) == ord('s'):
                    break
                if cv2.waitKey(10) == ord('q'):
                    return
        # bg_A = values[-22]
        # print('BG Size', len(bg_A))
        # for i in range(0, len(bg_A), 10):
        #     idx_src = bg_A[i]
        #     pts_src = (idx_src % 640, int(idx_src / 640))
        #     # pts_tgt = (idx_tgt % 640 + 640, int(idx_tgt / 640))
        #     # rand_color = tuple(np.random.randint(0, 255, 3).astype(int))
        #     colors = (0, 0, 0)
        #     colors = (int(colors[0]), int(colors[1]), int(colors[2]))
        #     # cv2.line(img_concate, pts_src, pts_tgt, colors)q
        #     cv2.circle(img_concate, pts_src, 2, colors, thickness=-1)  # Filled circle
        #     # cv2.circle(img_concate, pts_tgt, 2, colors, thickness=-1)  # Filled circle


# function for visualize testing
def indices1d_remap_new_uv(old_indices, old_shape, new_shape, shift=(0, 0)):  # Assume patch placed on left-upper
    """
    :param old_indices:  1d index list for old_shape
    :param old_shape:  h,w of new image
    :param new_shape: h,w of new image
    :param shift: Horizontal and vertical shift from (left-upper) origin (shift_hori,shift_vert)
    :return:
    """
    h0, w0, _ = old_shape
    h1, w1, _ = new_shape
    shift_w, shift_h = shift
    # Old to 2d ->
    idx_h, idx_w = old_indices // w0, old_indices % w0
    u, v = idx_w + shift_w, idx_h + shift_h
    return u, v


if __name__ == '__main__':
    static_dataset_vis_test()

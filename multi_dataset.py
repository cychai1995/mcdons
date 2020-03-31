import numpy as np
import os, sys

from PIL import Image  # CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
from torchvision import transforms
from torch.utils.data import Dataset
import cv2  # For testing code only
from sklearn.neighbors import NearestNeighbors
# from ipdb import set_trace


class MultiDataset(Dataset):
    def __init__(self, sampling_config, data_config, multi_static=False):
        # Access all parameters directly
        self.__dict__.update(sampling_config)
        self.__dict__.update(data_config)
        self.multi_static = multi_static
        transform_list = []
        if self.multi_static:
            self.color_jit = False
            self.randomize_background = False

        if self.color_jit:
            transform_list.extend([lambda x: Image.fromarray(x),
                                   transforms.ColorJitter(*tuple(self.color_jit_range))])
        transform_list.extend([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
        self.transform = transforms.Compose(transform_list)
        if self.randomize_background:
            self.texture_libs = []
            for fd in self.texture_dirs:
                folder = os.path.join(self.texture_base, fd)
                self.texture_libs.extend([os.path.join(folder, img) for img in os.listdir(folder)])
        self.cam_mat_inv = np.linalg.inv(self.cam_mat)

        if self.separate_match_non_match_sampling and self.A_A2_only:  # A,A2 only case, use self.match_classes to remap
            self.clf_remap_dict = {0: 0}
            self.clf_remap_dict.update({k: i + 1 for i, k in enumerate(self.match_classes)})
        else:
            self.clf_remap_dict = {0: 0}
            self.clf_remap_dict.update({k: i + 1 for i, k in enumerate(self.object_classes)})

    def __getitem__(self, idx):  # Sampling Dataset, one training triplet requires ~=400ms
        if self.multi_static:
            self.rs = np.random.RandomState(idx + 1234)
        else:
            self.rs = np.random.RandomState(
                os.getpid() + int.from_bytes(os.urandom(4), byteorder="little") >> 1)  # thread indep. random
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
        
        >>> Part1 Total Data Points <<<
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
                class_prob = self.class_match_sample_weight / np.sum(self.class_match_sample_weight)
                if self.separate_match_non_match_sampling:
                    match_class_set = self.rs.choice(self.match_classes,
                                                     size=self.matching_class_set_n,
                                                     p=class_prob,
                                                     replace=False)
                    non_match_exclude_set = list(set(self.non_match_classes) - set(match_class_set))
                    non_match_class_set = self.rs.permutation(non_match_exclude_set)[0:self.non_matching_class_set_n]
                else:
                    total_class = self.rs.choice(self.object_classes,
                                                 size=self.matching_class_set_n + self.non_matching_class_set_n,
                                                 p=class_prob,
                                                 replace=False)
                    match_class_set, non_match_class_set = total_class[0:self.matching_class_set_n], \
                                                           total_class[
                                                           self.matching_class_set_n:self.matching_class_set_n + \
                                                                                     self.non_matching_class_set_n]
                # Each set , each class sample one object id
                sample_list = lambda lst: self.rs.permutation(lst)[0]
                match_ids = list(map(sample_list, [self.class_to_object_ids[i] for i in match_class_set]))
                non_match_ids = list(map(sample_list, [self.class_to_object_ids[i] for i in non_match_class_set]))

                # Each object sample one folder(scene), result is list of [folder_name,N]
                match_folders = list(map(sample_list, [self.id_to_scenes[i] for i in match_ids]))
                non_match_folders = list(map(sample_list, [self.id_to_scenes[i] for i in non_match_ids]))

                # Each folder sample two image pairs. For matching pair, sampling range is +/-self.match_frame_near

                match_folder_ids = [(folder, self.sample_pair_in_range(n, self.match_frame_near)) for folder, n in
                                    match_folders]
                non_match_folder_ids = [(folder, self.rs.permutation(int(n))[0]) for folder, n in non_match_folders]

                # Save the object-id and class mapping
                class_mapping_dict = {o_id: cls for o_id, cls in zip(match_ids, match_class_set)}
                class_mapping_dict.update({o_id: cls for o_id, cls in zip(non_match_ids, non_match_class_set)})

                # img = cv2.imread(img_path)
                # mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
                # depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

                # [img_m11,img_m12,img_m21,img_m22,...], 2D image
                match_img = [self.open_img(folder, 'color', p, 0)
                             for folder, pair in match_folder_ids for p in pair]

                # [[mask2d11,mask2d12],[mask2d21,mask2d22],...], 2d mask
                match_mask = [[self.open_img(folder, 'label', pair[0], 1),
                               self.open_img(folder, 'label', pair[1], 1)] \
                              for folder, pair in match_folder_ids]
                # 2 match images as default background for A,A2
                bk_A_default = match_img[0].copy()
                bk_A2_default = match_img[1].copy()
                bk_mask_A, bk_mask_A2 = match_mask[0][0].copy(), match_mask[0][1].copy()

                # [mask11,mask12,mask21,mask22,...], 1d mask
                match_mask_1d = [np.where(mask2d.reshape(-1) == obj_id)[0] for masks, obj_id in
                                 zip(match_mask, match_ids) for mask2d in masks]
                match_bg_1d = [np.where(mask2d.reshape(-1) == 0)[0] for masks in match_mask for mask2d in masks]

                # [depth11,depth12,depth21,depth22,...], 2D depth
                match_depth = [self.open_img(folder, 'depth', p, 1)
                               for folder, pair in match_folder_ids for p in pair]
                match_transforms = [self.relative_transform(folder, pair[0], pair[1]) for folder, pair in
                                    match_folder_ids]

                non_match_img = [self.open_img(folder, 'color', idx, 0)
                                 for folder, idx in non_match_folder_ids]
                non_match_mask = [self.open_img(folder, 'label', idx, 1) \
                                  for folder, idx in non_match_folder_ids]
                # 1 non-match image as default background B
                bk_B_default = non_match_img[0].copy()
                bk_mask_B = non_match_mask[0].copy()

                non_match_mask_1d = [np.where(mask2d.reshape(-1) == obj_id)[0] for mask2d, obj_id in
                                     zip(non_match_mask, non_match_ids)]
                non_match_bg_1d = [np.where(mask2d.reshape(-1) == 0)[0] for mask2d in non_match_mask]

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
                for i in range(len(match_img)):
                    """
                     crop_contour_patch(img, mask_2d, tracking_1d_indices) ->
                      img, mask_1d, bg_1d, mask_2d, remap_indices, keep_indices
                    """
                    if i % 2 == 0:  # Source
                        match_img[i], match_mask_1d[i], match_bg_1d[i], match_mask[i // 2][
                            i % 2], indices, keep_indices = \
                            crop_contour_patch(match_img[i], match_mask[i // 2][i % 2], [match_src_indices[i // 2]],
                                               match_ids[i // 2])
                        match_src_indices[i // 2] = indices[0]
                        match_tgt_indices[i // 2] = match_tgt_indices[i // 2][keep_indices[0]]
                    else:  # i%2==1, Target
                        match_img[i], match_mask_1d[i], match_bg_1d[i], match_mask[i // 2][
                            i % 2], indices, keep_indices = \
                            crop_contour_patch(match_img[i], match_mask[i // 2][i % 2], [match_tgt_indices[i // 2]],
                                               match_ids[i // 2])
                        match_tgt_indices[i // 2] = indices[0]
                        match_src_indices[i // 2] = match_src_indices[i // 2][keep_indices[0]]

                # Image B (Non-match) processing
                for i in range(len(non_match_img)):
                    non_match_img[i], non_match_mask_1d[i], non_match_bg_1d[i], non_match_mask[i], _, _ = \
                        crop_contour_patch(non_match_img[i], non_match_mask[i], None, non_match_ids[i])
                # TODO: Move RSR Resample error by resample rescale/angle factor only (try times=5) =>Useless
                # Random Scale Rotate Part for matching pairs

                # rsr_retry_time = 0
                # while rsr_retry_time < 5:
                if self.random_scale_rotate:
                    for i in range(len(match_img)):
                        # Process Match src
                        resample = self.rs.randint(0, 3)  # cv2 version
                        scale = self.rs.uniform(self.scale_range_min, self.scale_range_max)
                        angle = self.rs.uniform(-self.angle_range, self.angle_range)
                        translation = self.rs.uniform(-self.translation_range, self.translation_range, 2)
                        if i % 2 == 0:  # src
                            match_img[i], match_keep_idx, match_new_indices, match_mask_1d[i], match_bg_1d[i] \
                                = scale_rotation_translation(
                                match_img[i],
                                [match_src_indices[i // 2]],
                                match_mask_1d[i],
                                match_bg_1d[i],
                                scale,
                                resample,
                                angle,
                                translation)
                            match_src_indices[i // 2] = match_new_indices[0]
                            match_tgt_indices[i // 2] = match_tgt_indices[i // 2][match_keep_idx[0]]
                        else:  # tgt
                            match_img[i], match_keep_idx, match_new_indices, match_mask_1d[i], match_bg_1d[i] \
                                = scale_rotation_translation(
                                match_img[i],
                                [match_tgt_indices[i // 2]],
                                match_mask_1d[i],
                                match_bg_1d[i],
                                scale,
                                resample,
                                angle,
                                translation)
                            match_tgt_indices[i // 2] = match_new_indices[0]
                            match_src_indices[i // 2] = match_src_indices[i // 2][match_keep_idx[0]]
                    # Non-match img
                    for i in range(len(non_match_img)):
                        resample = self.rs.randint(0, 3)  # cv2 version
                        scale = self.rs.uniform(self.scale_range_min, self.scale_range_max)
                        angle = self.rs.uniform(-self.angle_range, self.angle_range)
                        translation = self.rs.uniform(-self.translation_range, self.translation_range, 2)
                        non_match_img[i], _, _, non_match_mask_1d[i], non_match_bg_1d[i] \
                            = scale_rotation_translation(
                            non_match_img[i],
                            None,
                            non_match_mask_1d[i],
                            non_match_bg_1d[i],
                            scale,
                            resample,
                            angle,
                            translation)

                # Allocate space for each patch
                src_images = [match_img[i] for i in range(0, len(match_img), 2)]
                src_bg_masks = [match_bg_1d[i] for i in range(0, len(match_bg_1d), 2)]
                src_match_masks = [np.asarray(match_mask_1d[i]) for i in range(0, len(match_mask_1d), 2)]

                tgt_images = [match_img[i] for i in range(1, len(match_img), 2)]
                tgt_bg_masks = [match_bg_1d[i] for i in range(1, len(match_bg_1d), 2)]
                tgt_match_masks = [np.asarray(match_mask_1d[i]) for i in range(1, len(match_mask_1d), 2)]

                # cv2_show_img_lists(src_images)
                shifts_src = self.sample_patch_shifts_v2(src_images, (480, 640, 3))
                shifts_tgt = self.sample_patch_shifts_v2(tgt_images, (480, 640, 3))

                # Merge Match patches
                try:
                    img_A, canvas_label_A, keep_indices_list, remap_indices = merge_patches((480, 640, 3),
                                                                                            src_images,
                                                                                            shifts_src,
                                                                                            src_bg_masks,
                                                                                            src_match_masks,
                                                                                            [[m] for m in
                                                                                             match_src_indices],
                                                                                            match_ids)
                    match_src_indices = [m[0] for m in remap_indices]
                    match_tgt_indices = [indices[keep[0]] for indices, keep in
                                         zip(match_tgt_indices, keep_indices_list)]  # Filter Target indices
                    img_A2, canvas_label_A2, keep_indices_list, remap_indices = merge_patches((480, 640, 3),
                                                                                              tgt_images,
                                                                                              shifts_tgt,
                                                                                              tgt_bg_masks,
                                                                                              tgt_match_masks,
                                                                                              [[m] for m in
                                                                                               match_tgt_indices],
                                                                                              match_ids)
                    match_tgt_indices = [m[0] for m in remap_indices]
                    match_src_indices = [indices[keep[0]] for indices, keep in
                                         zip(match_src_indices, keep_indices_list)]  # Filter source indices

                    # Merge Non-match patches
                    shifts_non_match = self.sample_patch_shifts_v2(non_match_img, (480, 640, 3))
                    img_B, canvas_label_B, _, _ = merge_patches((480, 640, 3), non_match_img, shifts_non_match,
                                                                non_match_bg_1d, non_match_mask_1d,
                                                                None, non_match_ids)
                except IndexError as e:  # IndexedError
                    print('Fatal: Merge patch Index Error, resample')
                    chk_match = 0
                    continue

                chk_match = np.min([s.shape[0] for s in match_src_indices])
                # if chk_match == 0:
                #         rsr_retry_time += 1
                #         print('One of the match is 0 after RSR, retry=', rsr_retry_time)
                #         continue
                #     else:
                #         break

                if chk_match == 0:
                    print('One of the match is 0 after RSR, resample')
                    continue

            # TODO: tracking unique id -> another dataset
            if self.randomize_background:
                img_paths = [self.texture_libs[idx] for idx in self.rs.randint(0, len(self.texture_libs), 3)]
            dsize = (img_A.shape[1], img_A.shape[0])
            enable = self.rs.uniform(0, 1, 3)
            kernel_dil = np.ones((15, 15), np.uint8)
            mask = canvas_label_A == 0
            if enable[0] < self.bk_random_prob and self.randomize_background:
                img_A[mask] = cv2.resize(cv2.imread(img_paths[0]), dsize, interpolation=cv2.INTER_CUBIC)[mask]
            else:
                bk_mask_A = cv2.dilate(bk_mask_A, kernel_dil, iterations=3)
                bk_A_default[bk_mask_A != 0] = np.mean(bk_A_default.reshape(-1, 3), axis=0)
                img_A[mask] = bk_A_default[mask]

            mask = canvas_label_A2 == 0
            if enable[1] < self.bk_random_prob and self.randomize_background:
                img_A2[mask] = cv2.resize(cv2.imread(img_paths[1]), dsize, interpolation=cv2.INTER_CUBIC)[mask]
            else:
                bk_mask_A2 = cv2.dilate(bk_mask_A2, kernel_dil, iterations=3)
                bk_A2_default[bk_mask_A2 != 0] = np.mean(bk_A2_default.reshape(-1, 3), axis=0)
                img_A2[mask] = bk_A2_default[mask]

            mask = canvas_label_B == 0
            if enable[2] < self.bk_random_prob and self.randomize_background:
                img_B[mask] = cv2.resize(cv2.imread(img_paths[2]), dsize, interpolation=cv2.INTER_CUBIC)[mask]
            else:
                bk_mask_B = cv2.dilate(bk_mask_B, kernel_dil, iterations=3)
                bk_B_default[bk_mask_B != 0] = np.mean(bk_B_default.reshape(-1, 3), axis=0)
                img_B[mask] = bk_B_default[mask]

                # BGR -> RGB for images (CV2)

            # Match result for sampling
            # All id is object id
            match_idx_A = np.hstack(match_src_indices)
            match_idx_A2 = np.hstack(match_tgt_indices)
            match_class_id = np.repeat(match_ids, [len(m) for m in match_src_indices])

            # mask_A_1d = np.where(canvas_label_A.reshape(-1) != 0)[0]  # Expanded border mask
            # fg_A_id = canvas_label_A[canvas_label_A != 0]
            #
            # mask_A2_1d = np.where(canvas_label_A2.reshape(-1) != 0)[0]  # Expanded border mask
            # fg_A2_id = canvas_label_A2[canvas_label_A2 != 0]
            #
            # mask_B_1d = np.where(canvas_label_B.reshape(-1) != 0)[0]  # Expanded border mask
            # fg_B_id = canvas_label_B[canvas_label_B != 0]

            mask_A_1d = np.where((canvas_label_A.reshape(-1) != 0) & (canvas_label_A.reshape(-1) != 255))[0]

            fg_A_id = canvas_label_A[(canvas_label_A != 0) & (canvas_label_A != 255)]

            mask_A2_1d = np.where((canvas_label_A2.reshape(-1) != 0) & (canvas_label_A2.reshape(-1) != 255))[0]
            fg_A2_id = canvas_label_A2[(canvas_label_A2 != 0) & (canvas_label_A2 != 255)]

            mask_B_1d = np.where((canvas_label_B.reshape(-1) != 0) & (canvas_label_B.reshape(-1) != 255))[
                0]  # Expanded border mask
            fg_B_id = canvas_label_B[(canvas_label_B != 0) & (canvas_label_B != 255)]

            fg_A_cls, fg_A2_cls, fg_B_cls = self.remap_label_lists([fg_A_id, fg_A2_id, fg_B_id], class_mapping_dict)

            bg_A_1d = np.where(canvas_label_A.reshape(-1) == 0)[0]
            bg_A2_1d = np.where(canvas_label_A2.reshape(-1) == 0)[0]
            bg_B_1d = np.where(canvas_label_B.reshape(-1) == 0)[0]

            """
            >>> Part2 Sample Data Points <<<
            Positive, Negative....
            """
            # For match -> As much as possible, not balance
            if self.balanced_match:
                s_match_idx_A, s_match_idx_A2 = self.sample_balanced_pairs(match_class_id, match_idx_A, match_idx_A2,
                                                                           self.match_n)
                s_match_hn_id, s_match_hn_A, s_match_hn_A2 = self.sample_balanced_pairs(match_class_id, match_idx_A,
                                                                                           match_idx_A2,
                                                                                           self.hard_negative_match,
                                                                                           return_label=True)
                s_match_hn_label=self.remap_label_lists([s_match_hn_id], class_mapping_dict)[0]
            else:
                # TODO: s_match_hn_cls
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

            s_fg_B = self.sample_balanced_elements(fg_B_cls, mask_B_1d, self.background_n_per_class)
            s_bg_B = self.sample_elements_1d(bg_B_1d, len(s_fg_B))

            # Background Same (_background_same)
            s_bg_A_same = self.sample_elements_1d(bg_A_1d, self.background_same_n)
            s_bg_A2_same = self.sample_elements_1d(bg_A2_1d, self.background_same_n)
            s_bg_B_same = self.sample_elements_1d(bg_B_1d, self.background_same_n)

            # For closest pixel evaluation(_match_closest_pixel_distance)
            s_match_pix_A, s_match_pix_A2 = self.sample_balanced_pairs(match_class_id, match_idx_A, match_idx_A2,
                                                                       self.pix_dist_match_n)
            # For classification(mask)
            s_cls_label_A, s_cls_A, balance_N = self.sample_balanced_elements(fg_A_cls, mask_A_1d,
                                                                              self.segmentation_n // 2,
                                                                              return_label=True, return_N=True)
            s_cls_label_A2, s_cls_A2, balance_N2 = self.sample_balanced_elements(fg_A2_cls, mask_A2_1d, balance_N,
                                                                                 return_label=True, return_N=True)
            balance_N = min(balance_N, balance_N2)
            s_cls_label_B, s_cls_B = self.sample_balanced_elements(fg_B_cls, mask_B_1d, balance_N,
                                                                   return_label=True)
            s_cls_bk_A = self.sample_elements_1d(bg_A_1d, balance_N)

            # s_cls_labels = np.hstack((s_cls_label_A, s_cls_label_A2, np.zeros_like(s_cls_bk_A), s_cls_label_B))
            # Mapping class label from class_id to 0~N_classes that actually used in the sampling config, for segmentation prediction
            if self.A_A2_only:
                s_cls_labels = np.hstack((s_cls_label_A, s_cls_label_A2, np.zeros_like(s_cls_bk_A)))
                s_cls_labels = self.remap_label_lists([s_cls_labels], self.clf_remap_dict)[0]
            else:
                s_cls_labels = np.hstack((s_cls_label_A, s_cls_label_A2, np.zeros_like(s_cls_bk_A), s_cls_label_B))
                s_cls_labels = self.remap_label_lists([s_cls_labels], self.clf_remap_dict)[0]

            # Sample mask for cross non_match (_contrastive_closest_non_matchAB)
            hn_sample_n = np.min([len(mask_A_1d), len(mask_A2_1d), len(mask_B_1d), self.hard_negative_AB_n])
            s_mask_A_hn = self.sample_elements_1d(mask_A_1d, hn_sample_n)
            s_mask_A2_hn = self.sample_elements_1d(mask_A2_1d, hn_sample_n)
            # Sample mask for _contrastive_closest_non_matchAA2, too big mask will use too many memory
            s_mask_A_hn_AA2, keep = self.sample_elements_1d(mask_A_1d, self.mask_max_hard_negative, return_label=True)
            s_mask_A_hn_AA2_cls = fg_A_cls[keep]
            s_mask_A2_hn_AA2, keep = self.sample_elements_1d(mask_A2_1d, self.mask_max_hard_negative, return_label=True)
            s_mask_A2_hn_AA2_cls = fg_A2_cls[keep]

            # BGR -> RGB
            img_A = img_A[:, :, [2, 1, 0]]
            img_A2 = img_A2[:, :, [2, 1, 0]]
            img_B = img_B[:, :, [2, 1, 0]]
            # Transform
            img_A, img_A2, img_B = tuple(map(self.transform, [img_A, img_A2, img_B]))

            return_vals = (img_A, img_A2, img_B,
                           mask_A_1d, mask_A2_1d, mask_B_1d, \
                           fg_A_cls, fg_A2_cls, fg_B_cls, \
                           bg_A_1d, bg_A2_1d, bg_B_1d, \
                           match_class_set, non_match_class_set, \
                           s_match_idx_A, s_match_idx_A2, s_match_hn_A, s_match_hn_A2, \
                           s_mask_A_hn_AA2, s_mask_A2_hn_AA2, \
                           s_fg_A, s_bg_A, s_fg_A2, s_bg_A2, s_fg_B, s_bg_B, \
                           s_bg_A_same, s_bg_A2_same, s_bg_B_same, \
                           s_match_pix_A, s_match_pix_A2, \
                           s_mask_A_hn, s_mask_A2_hn, \
                           s_match_hn_label, s_mask_A_hn_AA2_cls, s_mask_A2_hn_AA2_cls, \
                           s_cls_A, s_cls_A2, s_cls_B, s_cls_bk_A, s_cls_labels)
            if self.return_checker(return_vals):
                break
            else:
                print('Dataset Re-sample, Return Checker Fails')
        return return_vals

    @staticmethod
    def remap_label_lists(id_list, mapping):
        """
        :param id_list: list of 1d ndarray
        :param mapping: dictionary of m
        :return:
        """
        return [replace_with_dict(l, mapping) for l in id_list]

    def sample_balanced_elements(self, class_label, elements, N_per_class, return_label=False, return_N=False):
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
            if return_N:
                return np.repeat(labels, sample_n), s_elements, sample_n
            else:
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

    @staticmethod
    def calculate_non_match_expand(match_idx_A, match_idx_A2, mask_A_1d, mask_A2_1d, canvas_size, kernel_size):
        """
        :param match_idx_A:
        :param match_idx_A2:
        :param mask_A_1d:
        :param mask_A2_1d:
        :param canvas_size: (h,w)
        :param kernel_size: for expanding the match pixel area location
        :return:
        """
        match_mask_A = np.zeros(canvas_size, dtype=np.uint8).reshape(-1)
        match_mask_A2 = np.zeros(canvas_size, dtype=np.uint8).reshape(-1)
        match_mask_A[match_idx_A] = 1
        match_mask_A2[match_idx_A2] = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expand_mask_A = cv2.dilate(match_mask_A.reshape(canvas_size), kernel, iterations=1)
        expand_mask_A2 = cv2.dilate(match_mask_A2.reshape(canvas_size), kernel, iterations=1)
        expand_idx_A = np.where(expand_mask_A.reshape(-1) == 1)[0]
        expand_idx_A2 = np.where(expand_mask_A2.reshape(-1) == 1)[0]
        non_match_A_1d = np.setdiff1d(mask_A_1d, expand_idx_A)
        non_match_A2_1d = np.setdiff1d(mask_A2_1d, expand_idx_A2)
        return non_match_A_1d, non_match_A2_1d

    def sample_elements_1d(self, elements, N, return_label=False):
        N_sample = min(len(elements), N)
        idx = random_permute_derangement(len(elements), self.rs)
        if return_label:
            return elements[idx[0:N_sample]], idx[0:N_sample]
        else:
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

    # def sample_patch_shiftsV0(self, patches, canvas_shape):
    #     N = len(patches)
    #     h, w, _ = canvas_shape
    #     max_ph, max_pw, _ = np.max(np.vstack([patch.shape for patch in patches]), axis=0)
    #     noise_max_u = min(self.noise_max_u, (w - max_pw) // 2)
    #     noise_max_v = min(self.noise_max_v, (h - max_ph) // 2)
    #     # print(noise_max_u, noise_max_v)
    #     v_max, u_max = h - max_ph - noise_max_v, w - max_pw - noise_max_u
    #     v_min, u_min = noise_max_v, noise_max_u
    #     u_max = u_max + 1 if u_max == u_min else u_max
    #     v_max = v_max + 1 if v_max == v_min else v_max
    #     # print(u_min, u_max, v_min, v_max)
    #     if N == 3:  # Triangle shape
    #         u_step = (np.linspace(u_min, u_max, 3) + self.rs.randint(-noise_max_u, noise_max_u + 1, N)).astype(
    #             np.int32)
    #         if self.rs.uniform(0, 1, 1) < 0.5:
    #             v_step = np.array([v_min, v_max, v_min]) + self.rs.randint(-noise_max_v, noise_max_v + 1, N)
    #         else:
    #             v_step = np.array([v_max, v_min, v_max]) + self.rs.randint(-noise_max_v, noise_max_v + 1, N)
    #     elif N == 4:  # Window shape
    #         u_step = np.array([u_min, u_min, u_max, u_max]) + self.rs.randint(-noise_max_u, noise_max_u + 1, N)
    #         v_step = np.array([v_min, v_max, v_min, v_max]) + self.rs.randint(-noise_max_v, noise_max_v + 1, N)
    #     else:
    #         u_step = self.rs.randint(u_min, u_max, N) + self.rs.randint(-noise_max_u, noise_max_u + 1, N)
    #         v_step = self.rs.randint(v_min, v_max, N) + self.rs.randint(-noise_max_v, noise_max_v + 1, N)
    #     # Generate list of shift tuple
    #     return [(u, v) for u, v in zip(u_step, v_step)]

    def sample_patch_shifts(self, patches, canvas_shape):
        # print('v1')
        N = len(patches)
        h, w, _ = canvas_shape
        shifts = []
        ut_space = [max(1, (w // N - p.shape[1])) for p in patches]
        vt_space = [max(1, (h // N - p.shape[0])) for p in patches]
        rand_seq_u = self.rs.permutation(N)
        rand_seq_v = self.rs.permutation(N)
        w_sp, h_sp = w // N, h // N
        for i, p in enumerate(patches):
            u = min(rand_seq_u[i] * w_sp + self.rs.randint(0, ut_space[i]), w - p.shape[1])
            v = min(rand_seq_v[i] * h_sp + self.rs.randint(0, vt_space[i]), h - p.shape[0])
            shifts.append([u, v])
        return shifts

    def sample_patch_shifts_v2(self, patches, canvas_shape, rev=True):
        # print('v2')
        N = len(patches)
        h, w, _ = canvas_shape
        shifts = np.zeros((N, 2), dtype=np.int)
        bounds = np.zeros((N, 2), dtype=np.int)
        tree = PackNode((w, h))  # uv space
        areas = [p.shape[0] * p.shape[1] for p in patches]

        # rev=False
        sort_idx = sorted(range(N), key=lambda k: areas[k], reverse=rev)
        # image = np.zeros((480, 640, 3), dtype=np.uint8)  # debug
        for s in sort_idx:
            uv = tree.insert((patches[s].shape[1], patches[s].shape[0]))
            if uv is None:
                if not rev:
                    # print('Pack size too small. Return pure random shift')
                    return self.sample_patch_shifts(patches, canvas_shape)
                else:
                    return self.sample_patch_shifts_v2(patches, canvas_shape, rev=False)
            else:
                # print(uv.area)
                u1, v1, u2, v2 = uv.area
                shifts[s] = [u1, v1]
                bounds[s] = [u2, v2]
                # image[v1:v2, u1:u2] = patches[s]  # debug
        # cv2_show_img_lists([image], 'Image 1')  # debug

        # image2 = np.zeros((480, 640, 3), dtype=np.uint8)  # debug

        # Calculate total random shifts available for u and v
        u_quota = w - np.max(bounds[:, 0])
        v_quota = h - np.max(bounds[:, 1])
        u_rand = self.rs.randint(0, 1000, N + 1)  # 1 more for supporting single object
        v_rand = self.rs.randint(0, 1000, N + 1)
        u_rand = (u_rand / (np.sum(u_rand) / u_quota)).astype(np.int)
        v_rand = (v_rand / (np.sum(v_rand) / v_quota)).astype(np.int)
        shifts[:, 0] += np.cumsum(u_rand[:-1])
        shifts[:, 1] += np.cumsum(v_rand[:-1])
        # for s in sort_idx: #debug
        #     h, w, _ = patches[s].shape
        #     ws, hs = shifts[s]
        # image2[hs:hs + h, ws:ws + w,:] = patches[s]
        # cv2_show_img_lists([image2], 'Image 2')  # debug
        return shifts

    @staticmethod
    def point_id_match(src_point_id, tgt_point_id, obj_model_points, distance_check):
        """
        :param src_point_id: H,W, -1 for noise point
        :param tgt_point_id: H,W, -1 for noise point
        :param obj_model_points: N,3 denoise points, global coordinate
        :param distance_check: unit:m
        :return:
        """
        src_valid_indices = np.where(src_point_id.reshape(-1) != -1)[0]
        tgt_valid_indices = np.where(tgt_point_id.reshape(-1) != -1)[0]

        src_points = obj_model_points[src_point_id.reshape(-1)[src_valid_indices], :]  # N,3
        tgt_points = obj_model_points[tgt_point_id.reshape(-1)[tgt_valid_indices], :]  # N,3
        # Violent NN Search
        dist, indices = find_NN(src_points, tgt_points, k=1)  # indices of src points
        keep = np.where(dist <= distance_check)[0]
        print('Keep:', len(keep))
        src_indices = src_valid_indices[indices[keep]]
        tgt_indices = tgt_valid_indices[keep]

        return src_indices, tgt_indices  # 1D index in image

    def sample_pair_in_range(self, N, near_range):
        N = int(N)
        n1 = self.rs.randint(N)
        while True:
            n2 = self.rs.randint(n1 - near_range, n1 + near_range) % N
            if n2 != n1:
                break
        return [n1, n2]

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
                    if isinstance(k, np.ndarray):
                        if k.shape[0] == 0:
                            return False
        return True


# http://code.activestate.com/recipes/442299/
class PackNode(object):
    """
    Creates an area which can recursively pack other areas of smaller sizes into itself.
    """

    def __init__(self, area):
        # if tuple contains two elements, assume they are width and height, and origin is (0,0)
        if len(area) == 2:
            area = (0, 0, area[0], area[1])
        self.area = area

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, str(self.area))

    def get_width(self):
        return self.area[2] - self.area[0]

    width = property(fget=get_width)

    def get_height(self):
        return self.area[3] - self.area[1]

    height = property(fget=get_height)

    def insert(self, area):
        if hasattr(self, 'child'):
            a = self.child[0].insert(area)
            if a is None: return self.child[1].insert(area)
            return a

        area = PackNode(area)
        if area.width <= self.width and area.height <= self.height:
            self.child = [None, None]
            self.child[0] = PackNode(
                (self.area[0] + area.width, self.area[1], self.area[2], self.area[1] + area.height))
            self.child[1] = PackNode((self.area[0], self.area[1] + area.height, self.area[2], self.area[3]))
            return PackNode((self.area[0], self.area[1], self.area[0] + area.width, self.area[1] + area.height))


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


def merge_patches(canvas_shape, patches, patch_shifts, bg_1d_masks, match_1d_masks, indices_1d_lists, layer_ids=None,
                  bypass_bg=True):
    """
    :param canvas_shape: tuple of the shape, eg (h,w,3)
    :param patches: list of image arrays
    :param patch_shifts: list of 2d coordinate shifts, list of tuple (u_shift,v_shift),
    :param bg_1d_masks: mask of background of patches
    :param match_1d_masks: mask of foreground object (Exclude expanded border)
    :param indices_1d_lists: list(patch) of list(each patch, several 1d indices to keep) of indices
    :param layer_ids: list of ids (int type) for the returned layer map (0 is background by default),len(layer_ids) == len(patches)
    :param bypass_bg: Patches are merged by the pasting rectangular area onto canvas. bypass_bg=True
                      to bypass pasting background areas onto the canvas, which keeps more matching points.
    :return: layer_label,layer_img
    """
    # patch_shift for each patch with shape (ph,pw) should lies inside the final image coordinate canvas_shape
    # No check here
    h, w, _ = canvas_shape
    layer_label = np.zeros((h, w), dtype=np.int32)  # 0 as background by default
    layer_img = np.zeros((h, w, 3), dtype=np.uint8)
    layer_ids = np.arange(len(patches)) + 1 if layer_ids is None else layer_ids  # label from 1
    # Step1: Paste and override
    for patch, shift, bg_mask, match_mask, id in zip(patches, patch_shifts, bg_1d_masks, match_1d_masks, layer_ids):
        ph, pw, _ = patch.shape
        if bypass_bg:
            # Only paste non_bg area (expanded object areas)
            patch_mask = np.ones((ph, pw)).reshape(-1) * 255
            patch_mask[bg_mask] = 0
            patch_mask[match_mask] = 1
            mask_fg = patch_mask.reshape((ph, pw))
            interest_label = layer_label[shift[1]:shift[1] + ph, shift[0]:shift[0] + pw]
            interest_img = layer_img[shift[1]:shift[1] + ph, shift[0]:shift[0] + pw, :]
            # print(ph, pw,np.sum(np.ones_like(mask_fg)[mask_fg == 1]), mask_fg.shape, interest_label.shape, layer_label.shape,
            #       ph, pw, shift)
            interest_label[mask_fg == 1] = id
            interest_label[mask_fg == 255] = 255
            interest_img[mask_fg != 0, :] = patch[mask_fg != 0, :]
        else:
            layer_label[shift[1]:shift[1] + ph, shift[0]:shift[0] + pw] = id
            layer_img[shift[1]:shift[1] + ph, shift[0]:shift[0] + pw] = patch
    # Step2: Remap layer indices after merge
    # Find label to keep matching indices
    keep_indices_list = []  # for store another list of 1d indices
    remap_indices_list = []
    if indices_1d_lists is not None:
        for patch, shift, idx_lists, id in zip(patches, patch_shifts, indices_1d_lists, layer_ids):
            ph, pw, _ = patch.shape
            interest_label = layer_label[shift[1]:shift[1] + ph, shift[0]:shift[0] + pw]
            keep_all = np.where(interest_label.reshape(-1) == id)[0]  # 1d indices of patch coordinate
            # idx_lists To UV
            local_remap = []
            local_keep = []
            for idx_1d in idx_lists:
                intersect, keep, _ = np.intersect1d(idx_1d, keep_all, return_indices=True)
                remap_indices = shift[0] + intersect % pw + (intersect // pw + shift[1]) * w
                local_remap.append(remap_indices)
                local_keep.append(keep)
            keep_indices_list.append(local_keep)
            remap_indices_list.append(local_remap)

    return layer_img, layer_label, keep_indices_list, remap_indices_list


def scale_rotation_translation(img, tracking_indices_list, mask_1d, bg_1d, scale, resample, rotation, translation,
                               rotate_protect=True, canvas_size=(480, 640)):
    """
    :param img: numpy array from cv2.imread (BGR format by default)
    :param tracking_indices_list: list of Indices list for tracking validity (e.g. [Match point indices A2, Match point indices AF] in 1D)
    :param mask_1d: Mask(Foreground) indices (e.g. Mask point index in 1D)
    :param bg_1d: Background indices (e.g. Background point index in 1D)
    :param scale: (0,1]
    :param resample: 0,1,2 for different interpolation mode, select randomly is preferred
    :param rotation: degree for counter clockwise rotation
    :param translation: (rw,rh) ratio of the rotated image's width and height. Suggested region for rw,rh is [-0.3,+0.3]
    :param rotate_protect: Prevent rotated patch size larger than img size

    :return:
         img_final: Final RGB image [mean pixel value is used for black region]
         keep_indices: List of the "indices of indices" that are still valid in img_final (for each array in tracking_indices_list)
         remap_indices: List of the new valid 1D-indices computed from tracking_indices_list
         idx_mask_new: New Foreground 1D-indices
         idx_bg_new: New Background 1D-indices
    """
    h, w, _ = img.shape
    mean_pixel = np.mean(img.reshape(-1, 3), axis=0)
    # img_final[:, :, 0:3] = 0  # 0,0,0
    keep_indices, remap_indices = [], []  # Dummy init
    if tracking_indices_list is not None:
        tracking_indices = [(idx_1d % w, idx_1d // w) for idx_1d in
                            tracking_indices_list]  # List of tuples of 1d (u,v) indices
    u_idx_mask, v_idx_mask = mask_1d % w, mask_1d // w
    u_idx_bg, v_idx_bg = bg_1d % w, bg_1d // w

    img_orig = img.copy()
    # Keep bg_1d points as background
    img_orig_bg = (np.ones((h, w), dtype=np.uint8) * 255)
    img_orig_bg[v_idx_bg, u_idx_bg] = 0

    img_orig_fg = np.zeros((h, w)).astype(np.uint8)
    img_orig_fg[v_idx_mask, u_idx_mask] = 255

    # Add mask
    # img_orig_bg.putalpha(Image.fromarray(alpha_mask_bg))  # For bg indices and final image
    # img_orig_fg.putalpha(Image.fromarray(alpha_mask_fg))  # For fg indices

    # *0 : Unmask for original version
    # img_final = np.zeros((h, w, 3), dtype=np.uint8)
    # bg_final = np.zeros((h, w), dtype=np.uint8)  # h,w
    # fg_final = np.zeros((h, w), dtype=np.uint8)  # h,w
    # img_final[:, :, 0:3] = mean_pixel

    h_new, w_new = np.round(np.array(img.shape[0:2]) * scale).astype(np.int64)
    resample_mode = [cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR][
        resample]  # TODO: Resample mode 0~4
    scaled_img, scaled_img_bg, scaled_img_fg = tuple(
        map(lambda x: cv2.resize(x, (w_new, h_new), interpolation=resample_mode),
            [img_orig, img_orig_bg, img_orig_fg]))

    # move scaled indices to center (for rotation)
    if tracking_indices_list is not None:
        tracking_indices = [(np.round(u * scale).astype(np.int64) - int(w_new / 2.0), np.round(v * scale).astype(
            np.int64) - int(h_new / 2.0)) for u, v in tracking_indices]
    # u_idx_mask, v_idx_mask = toCenter(u_idx_mask, v_idx_mask)
    # u_idx_bg, v_idx_bg = toCenter(u_idx_bg, v_idx_bg)

    # Image Rotation [Expanded rotation]
    rotated_img, rotated_img_bg, rotated_img_fg = rotate_bound(scaled_img, rotation, border_value=mean_pixel), \
                                                  rotate_bound(scaled_img_bg, rotation, border_value=0), \
                                                  rotate_bound(scaled_img_fg, rotation, border_value=0)

    # empty_pixel = np.all(rotated_img == np.array([0, 0, 0]), axis=2)  # Fill black with mean
    # rotated_img[empty_pixel] = 0  # 0,0,0
    h_rot, w_rot = rotated_img.shape[0:2]
    theta = np.deg2rad(rotation)
    # u=y,v=x
    if tracking_indices_list is not None:
        rotate_tracking_indices = [(np.round(np.sin(theta) * v + np.cos(theta) * u + int(w_rot / 2)).astype(
            np.int64), (np.cos(theta) * v - np.sin(theta) * u + int(h_rot / 2)).astype(np.int64)) for u, v in
            tracking_indices]

    # TODO: Resize the image if it's larger than the canvas after rotation
    if rotate_protect:
        scale_chk = max(rotated_img.shape[0] / canvas_size[0], rotated_img.shape[1] / canvas_size[1])
        if scale_chk > 1:
            # print('Trigger', scale_chk, img.shape, rotated_img.shape)
            rescale_factor = 1.0 / scale_chk
            h_new, w_new = (np.array(rotated_img.shape[0:2]) * rescale_factor).astype(np.int64)
            resample_mode = [cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR][
                resample]  # TODO: Resample mode 0~4
            rotated_img, rotated_img_bg, rotated_img_fg = tuple(
                map(lambda x: cv2.resize(x, (w_new, h_new), interpolation=resample_mode),
                    [rotated_img, rotated_img_bg, rotated_img_fg]))
            #
            # move scaled indices to center (for rotation)
            if tracking_indices_list is not None:
                rotate_tracking_indices = [
                    (np.round(u * rescale_factor).astype(np.int64), np.round(v * rescale_factor).astype(np.int64))
                    for u, v in rotate_tracking_indices]
        h_rot, w_rot = rotated_img.shape[0:2]
    # calculate translation pixels,
    rw, rh = translation
    trans_u, trans_v = int(w_rot * rw), int(h_rot * rh)

    # *1 Construct a coordinate system that has 2 rectangles
    # lu1 = (-int(w / 2.0), -int(h / 2.0))  # left-upper point for rect 1
    # lu2 = (-int(w_rot / 2.0) + trans_u, -int(h_rot / 2.0) + trans_v)  # Add translation here
    # cv2_show_img_lists([rotated_img], 'Rotated img')
    # r1, r2 = rectangle_intersect(lu1, w, h, lu2, w_rot, h_rot)
    # print('r1', r1, 'r2', r2)
    # Paste the image *
    # img_final[r1[1]:r1[3], r1[0]:r1[2], :] = rotated_img[r2[1]:r2[3], r2[0]:r2[2], :]
    # bg_final[r1[1]:r1[3], r1[0]:r1[2]] = rotated_img_bg[r2[1]:r2[3], r2[0]:r2[2]]
    # fg_final[r1[1]:r1[3], r1[0]:r1[2]] = rotated_img_fg[r2[1]:r2[3], r2[0]:r2[2]]

    # No cut and no translation version for *1 # Construct a coordinate system that has 2 rectangles *
    lu1 = (-int(w_rot / 2.0), -int(h_rot / 2.0))  # left-upper point for rect 1
    lu2 = (-int(w_rot / 2.0), -int(h_rot / 2.0))  # Add translation here
    # cv2_show_img_lists([rotated_img], 'Rotated img')
    # r1, r2 = rectangle_intersect(lu1, w_rot, h_rot, lu2, w_rot, h_rot)
    # print(lu1,lu2,r1,r2)
    r1, r2 = (0, 0, w_rot, h_rot), (0, 0, w_rot, h_rot)
    img_final = rotated_img
    bg_final = rotated_img_bg
    fg_final = rotated_img_fg

    # Filter out in-frame correspondence
    # keep_points_match = filterUV(u_idx_match_rot, v_idx_match_rot)
    if tracking_indices_list is not None:
        # *2
        # keep_indices = [np.intersect1d(np.intersect1d(np.where(u_rot >= r2[0])[0], \
        #                                               np.where(u_rot < r2[2])[0]), \
        #                                np.intersect1d(np.where(v_rot >= r2[1])[0], \
        #                                               np.where(v_rot < r2[3])[0])) \
        #                 for u_rot, v_rot in rotate_tracking_indices]

        # No cut version for *2
        keep_indices = [np.intersect1d(np.intersect1d(np.where(u_rot >= r2[0])[0], \
                                                      np.where(u_rot < r2[2])[0]), \
                                       np.intersect1d(np.where(v_rot >= r2[1])[0], \
                                                      np.where(v_rot < r2[3])[0])) \
                        for u_rot, v_rot in rotate_tracking_indices]

        # Remap to final frame
        # remapUV = lambda u_rot, v_rot, keep: (u_rot[keep] + lu2[0] - lu1[0], v_rot[keep] + lu2[1] - lu1[1])
        # u_idx_match_rot, v_idx_match_rot = remapUV(u_idx_match_rot, v_idx_match_rot, keep_points_match)
        # idx_match_new = v_idx_match_rot * w + u_idx_match_rot

        # *3
        # remap_indices = [u_rot[keep] + lu2[0] - lu1[0] + w * (v_rot[keep] + lu2[1] - lu1[1]) \
        #                  for (u_rot, v_rot), keep in zip(rotate_tracking_indices, keep_indices)]

        # No cut version for *3
        remap_indices = [u_rot[keep] + lu2[0] - lu1[0] + w_rot * (v_rot[keep] + lu2[1] - lu1[1]) \
                         for (u_rot, v_rot), keep in zip(rotate_tracking_indices, keep_indices)]

    # Calculate to 1d indices remapUV(u_idx_match_rot, v_idx_match_rot)

    # New background and mask1d calculation using alpha channel
    # idx_bg_new = np.where((bg_final[:, :, 3].reshape(-1)) == 0)[0]
    # idx_mask_new = np.where((fg_final[:, :, 3].reshape(-1)) == 255)[0]

    idx_bg_new = np.where((bg_final.reshape(-1)) <= 1)[0]
    idx_mask_new = np.where((fg_final.reshape(-1)) >= 254)[0]
    # Mean Pixel for other black points
    img_final[np.logical_and(bg_final <= 1, img_final.sum(axis=2) == 0), :] = mean_pixel

    """
    Without translation
    wrot_s, wrot_t = max(0, int((w_rot - w) / 2.0)), min(w_rot, int((w + w_rot) / 2.0))
    hrot_s, hrot_t = max(0, int((h_rot - h) / 2.0)), min(h_rot, int((h + h_rot) / 2.0))
    w_s, w_t = max(0, int((w - w_rot) / 2.0)), min(w, int((w + w_rot) / 2.0))
    h_s, h_t = max(0, int((h - h_rot) / 2.0)), min(h, int((h + h_rot) / 2.0))
    img_final[h_s:h_t, w_s:w_t] = rotated_img[hrot_s:hrot_t, wrot_s:wrot_t]
    """

    return img_final, keep_indices, remap_indices, idx_mask_new, idx_bg_new


def find_NN(base, query, k=1):
    """
    :param base:  N_base,D
    :param query: N_query,D
    :param k:
    :return:
    """
    nn = NearestNeighbors(n_neighbors=1, metric='l2', n_jobs=1)
    nn = nn.fit(base)
    dist, ind = nn.kneighbors(query, n_neighbors=k, return_distance=True)
    if k == 1:
        return dist[:, 0], ind[:, 0]
    else:
        return dist, ind


def contourArea(contour):
    cnt_box = cv2.boundingRect(contour)  # orig_row,orig_col,H,W
    return cnt_box[2] * cnt_box[3]  # Rect Area


def crop_contour_patch(img, mask_2d, tracking_1d_indices, obj_id):  # Including 255 areas(expanded border)
    """
    :param img: cv2 img
    :param mask_2d:
    :param tracking_1d_indices: list of tracking indices
    :return:img,new_mask_1d,new_bg_1d,new_indices_list, keep_indices
    """
    ret, thresh = cv2.threshold(mask_2d, 0, 255, cv2.THRESH_BINARY)  # Larger than 0 => 255 (Object + expanded boarder)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: contourArea(x), reverse=True)  # Largest one is the object patch
    # TODO: Double check patch mean
    w0, h0, width, height = cv2.boundingRect(contours[0])
    h, w, _ = img.shape  # Original Size
    img = img[h0:h0 + height, w0:w0 + width]
    remap_indices, keep_indices = [], []  # Dummy init
    if tracking_1d_indices is not None:
        tracking_indices = [(idx_1d % w - w0, idx_1d // w - h0) for idx_1d in
                            tracking_1d_indices]  # List of tuples of 1d (u,v) indices

        keep_indices = [np.intersect1d(np.intersect1d(np.where(u >= 0)[0], np.where(u < width)[0]), \
                                       np.intersect1d(np.where(v >= 0)[0], np.where(v < height)[0])) \
                        for u, v in tracking_indices]

        remap_indices = [u[keep] + width * (v[keep]) for (u, v), keep in zip(tracking_indices, keep_indices)]

    mask_2d = mask_2d[h0:h0 + height, w0:w0 + width]
    mask_1d = np.where(mask_2d.reshape(-1) == obj_id)[0]
    bg_1d = np.where(mask_2d.reshape(-1) == 0)[0]
    return img, mask_1d, bg_1d, mask_2d, remap_indices, keep_indices


# 1d to 2d


def rectangle_intersect(lu1, w1, h1, lu2, w2, h2):
    """
    https://stackoverflow.com/questions/4549544/

    :param lu1: Left upper 1 (u,v)
    :param w1:
    :param h1:
    :param lu2:  Left upper 2 (u,v)
    :param w2:
    :param h2:
    :return: intersection region of each rectangle image frame
    (us1,vs1,ut1,vt1),(us2,vs2,ut2,vt2)
    """
    u1, v1 = lu1
    u2, v2 = lu2
    left = max(u1, u2)
    right = min(u1 + w1, u2 + w2)
    top = max(v1, v2)
    bottom = min(v1 + h1, v2 + h2)
    assert left < right and bottom > top  # Intersection in image space
    if left == u1:
        us1 = 0
        us2 = left - u2
    else:
        us2 = 0
        us1 = left - u1

    if right == u1 + w1:
        ut1 = w1
        ut2 = right - u2
    else:
        ut2 = w2
        ut1 = right - u1

    if top == v1:
        vs1 = 0
        vs2 = top - v2
    else:
        vs2 = 0
        vs1 = top - v1

    if bottom == v1 + h1:
        vt1 = h1
        vt2 = bottom - v2
    else:
        vt2 = h2
        vt1 = bottom - v1

    return (us1, vs1, ut1, vt1), (us2, vs2, ut2, vt2)


def rotate_bound(image, angle, border_value=0):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2.0, h / 2.0)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate counter-clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos) + 0.5)
    nH = int((h * cos) + (w * sin) + 0.5)

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


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


def draw_match_test_multi():
    sample_n_for_display = 5000
    os.putenv('DISPLAY', ':0')
    import yaml
    with open('/home/chai/projects/visual_descriptor_learning/configs/1219_fpn_6d.yaml',
              'r') as f:
        dd = yaml.load(f)
    pd = MultiDataset(dd['sampling'], dd['test_data'])
    pd.transform = lambda x: x  # No normalization, for visualization
    pd.match_n = sample_n_for_display
    pd.background_n = 50000
    # keys = ['mask_A_1d', 'mask_A2_1d', 'mask_B_1d', \  2
    #         'fg_A_cls', 'fg_A2_cls', 'fg_B_cls', \ 5
    #         'match_class_set', 'non_match_class_set', \ 7
    #         's_match_idx_A', 's_match_idx_A2', 's_match_hn_A', 's_match_hn_A2', \ 11
    #         's_mask_A_hn_AA2', 's_mask_A2_hn_AA2', 13
    #         's_fg_A', 's_bg_A', 's_fg_A2', 's_bg_A2', 's_fg_B', 's_bg_B', \ 19
    #         's_bg_A_same', 's_bg_A2_same', 's_bg_B_same', \ 22
    #         's_match_pix_A', 's_match_pix_A2', \ 24
    #         's_mask_A_hn', 's_mask_A2_hn', \ 26
    #         's_cls_A', 's_cls_A2', 's_cls_B', 's_cls_bk_A', 's_cls_labels'] 31
    test_pairs = [[0, 1, 2], [3, 4, 5], [8, 9], [10, 11], [12, 13], [14, 16, 18], [15, 17, 19], [20, 21, 22], [23, 24],
                  [25, 26], [27, 28, 29]]

    while True:
        values = pd[0]
        # set_trace()
        img_A, img_A2, img_B, match_idx_src, match_idx_tgt = \
            values[0], values[1], values[2], values[11], values[12]  # A AF
        data = values[3:]

        print('Match N=', match_idx_src.shape[0])
        img_A_final, img_A2_final = np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 3), dtype=np.uint8)
        img_A_final[0:img_A.shape[0], 0:img_A.shape[1]] = img_A
        img_A2_final[0:img_A2.shape[0], 0:img_A2.shape[1]] = img_A2
        src_u, src_v = indices1d_remap_new_uv(match_idx_src, img_A.shape, (480, 640, 3))
        tgt_u, tgt_v = indices1d_remap_new_uv(match_idx_tgt, img_A2.shape, (480, 640, 3), shift=(640, 0))

        img_concate = np.concatenate((img_A_final, img_A2_final), axis=1)
        img_concate = np.ascontiguousarray(img_concate[:, :, [2, 1, 0]])
        sample = np.random.permutation(len(match_idx_src))[0:min(sample_n_for_display, len(match_idx_src))]
        print('Total Match=', len(match_idx_src), ' Sample length:', len(sample))
        # for cnt, (idx_src, idx_tgt) in enumerate(zip(match_idx_src, match_idx_tgt)):
        for i in range(0, len(sample), 100):
            pts_src = src_u[i], src_v[i]
            pts_tgt = tgt_u[i], tgt_v[i]
            # rand_color = tuple(np.random.randint(0, 255, 3).astype(int))
            colors = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            colors = (int(colors[0]), int(colors[1]), int(colors[2]))
            cv2.line(img_concate, pts_src, pts_tgt, colors)
            cv2.circle(img_concate, pts_src, 2, colors, thickness=-1)  # Filled circle
            cv2.circle(img_concate, pts_tgt, 2, colors, thickness=-1)  # Filled circle
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

        cv2.imshow("Test", img_concate)
        while True:
            if cv2.waitKey(10) == ord('s'):
                break
            if cv2.waitKey(10) == ord('q'):
                return


def multi_dataset_vis_test():
    disp_max = 500
    os.putenv('DISPLAY', ':0')
    import yaml
    with open('/home/chai/projects/visual_descriptor_learning/exps_final/exp1/others/0216_exp1_ours_443_MH_nbk.yaml',
              'r') as f:
        dd = yaml.load(f)
    pd = MultiDataset(dd['sampling'], dd['test_data'])
    # set_trace()
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
    test_pairs = [[0, 1, 2, False], [8, 9, True], [10, 11, True], [12, 13, False],
                  [14, 16, 18, False], [15, 17, 19, False],
                  [20, 21, 22, False], [23, 24, True],
                  [25, 26, False], [27, 28, 29, False]]

    while True:
        values = pd[0]
        # set_trace()
        img_A, img_A2, img_B, match_idx_src, match_idx_tgt = \
            values[0], values[1], values[2], values[11], values[12]  # A AF
        data = values[3:]

        print('Match N=', match_idx_src.shape[0])
        img_concate3 = np.concatenate((img_A, img_A2, img_B), axis=1)
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
    multi_dataset_vis_test()
    # disp_max = 5000
    # os.putenv('DISPLAY', ':0')
    # import yaml
    #
    # with open('/home/chai/projects/visual_descriptor_learning/exps_iros/others/0116_3c.yaml',
    #           'r') as f:
    #     dd = yaml.load(f)
    # pd = MultiDataset(dd['sampling'], dd['test_data'])
    # pd.transform = lambda x: x  # No normalization, for visualization
    # pd.match_n = disp_max
    # pd.background_n = 50000
    # for i in range(1000):
    #     print(i)
    #     aa = pd[0]

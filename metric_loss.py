import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace
import gc

"""
Metric Learning Loss
Does not support batch training currently
"""


class MetricLoss:
    def __init__(self, config, device):
        self.conf = config
        self.device = device
        self.cur_loss = {}

    def set_data(self, desp_A, desp_A2, desp_B, loader_values):
        loader2gpu = lambda key, val: self.__dict__.update({key: val[0].to(torch.long).to(self.device)})
        # The following lines copy from dataloader __getitem__(), except for the first three images
        keys = ['mask_A_1d', 'mask_A2_1d', 'mask_B_1d', \
                'fg_A_cls', 'fg_A2_cls', 'fg_B_cls', \
                'bg_A_1d', 'bg_A2_1d', 'bg_B_1d', \
                'match_class_set', 'non_match_class_set', \
                's_match_idx_A', 's_match_idx_A2', 's_match_hn_A', 's_match_hn_A2', \
                's_mask_A_hn_AA2', 's_mask_A2_hn_AA2',
                's_fg_A', 's_bg_A', 's_fg_A2', 's_bg_A2', 's_fg_B', 's_bg_B', \
                's_bg_A_same', 's_bg_A2_same', 's_bg_B_same', \
                's_match_pix_A', 's_match_pix_A2', \
                's_mask_A_hn', 's_mask_A2_hn', \
                's_match_hn_label', 's_mask_A_hn_AA2_cls', 's_mask_A2_hn_AA2_cls', \
                's_cls_A', 's_cls_A2', 's_cls_B', 's_cls_bk_A', 's_cls_labels']
        assert len(keys) == len(loader_values), "Please check the values returned from Dataloader"
        list(map(loader2gpu, keys, loader_values))  # remove batch dim
        # using self.XXX to access the data returned from dataloader with name XXX

        # desp_X: [1,3,480,640]
        self.img_width = desp_A.size(3)  # For pixel distance
        if self.conf.l2normalize:
            norm_a = desp_A.norm(p=2, dim=1)
            desp_A = desp_A.div(norm_a)
            norm_a2 = desp_A2.norm(p=2, dim=1)
            desp_A2 = desp_A2.div(norm_a2)
            if not self.conf.A_A2_only:
                norm_b = desp_B.norm(p=2, dim=1)
                desp_B = desp_B.div(norm_b)
        desp_dim = desp_A.size(1)
        desp_A = desp_A.view(desp_dim, -1).permute(1, 0)  # [307200,3]
        desp_A2 = desp_A2.view(desp_dim, -1).permute(1, 0)
        if not self.conf.A_A2_only:
            desp_B = desp_B.view(desp_dim, -1).permute(1, 0)

        self.desp_A = desp_A  # [307200,3]
        self.desp_A2 = desp_A2
        self.desp_B = desp_B

    def set_centers(self, net_center):
        self.centers = net_center
        self.get_all_clf_label()

    def get_loss_records(self):
        return self.cur_loss  # Only current loss

    def add_loss_record(self, key_list, tensor_list):
        for k, v in zip(key_list, tensor_list):
            self.cur_loss[k] = v.detach().cpu().float().item()

    """
    Loss Functions
    """

    def get_matching_loss(self, split=False):  # match and non-match and background
        match_loss = self._same_objects_match()
        objects_background = self._objects_background()
        background_same = self._background_same()
        if self.conf.contrastive_non_zero:
            match_loss = match_loss[match_loss > 0]
            objects_background = objects_background[objects_background > 0]
            background_same = background_same[background_same > 0]

        # if self.conf.contrastive_non_zero:
        #     match_loss = match_loss[match_loss > 0]
        #     non_match_cross = non_match_cross[non_match_cross > 0]
        #     non_match_diff = non_match_diff[non_match_diff > 0]
        #     non_match_same = non_match_same[non_match_same > 0]
        #     objects_background = objects_background[objects_background > 0]

        if self.conf.contrastive_sum_loss:
            match_loss = match_loss.sum()
            objects_background = objects_background.sum()
            background_same = background_same.sum()
        else:
            match_loss = self.tensor_mean(match_loss.float())
            objects_background = self.tensor_mean(objects_background.float())
            background_same = self.tensor_mean(background_same.float())

        self.add_loss_record(
            ['match_loss', 'objects_background',
             'background_same'],
            [match_loss, objects_background, background_same])
        if split:
            return match_loss, objects_background, background_same
        else:
            return match_loss + objects_background + background_same

    def get_hard_negative_loss(self):
        if not self.conf.A_A2_only:
            if self.conf.contrastive_sum_loss:
                hard_AA2 = self._contrastive_closest_non_matchAA2().sum()
                hard_AB = self._contrastive_closest_non_matchAB().sum()
            else:
                hard_AA2 = self.tensor_mean(self._contrastive_closest_non_matchAA2().float())
                hard_AB = self.tensor_mean(self._contrastive_closest_non_matchAB().float())
            self.add_loss_record(['hard_AA2', 'hard_AB'], [hard_AA2, hard_AB])
            return hard_AA2 + hard_AB
        else:
            if self.conf.contrastive_sum_loss:
                hard_AA2 = self._contrastive_closest_non_matchAA2().sum()
            else:
                hard_AA2 = self.tensor_mean(self._contrastive_closest_non_matchAA2().float())
            self.add_loss_record(['hard_AA2'], [hard_AA2])
            return hard_AA2

    def _same_objects_match(self):
        desp_pts_A = self.desp_A[self.s_match_idx_A]
        desp_pts_A2 = self.desp_A2[self.s_match_idx_A2]
        loss_map = F.relu((desp_pts_A - desp_pts_A2).norm(p=2, dim=1) - self.conf.match_m).pow(2)
        # if self.conf.contrastive_non_zero:
        #     loss_map = loss_map[loss_map > 0]
        if loss_map.size(0) == 0:
            return self.np2torch([0.0])
        return loss_map

    def _objects_background(self):
        desp_pts_fg_A = self.desp_A[self.s_fg_A]
        desp_pts_bg_A = self.desp_A[self.s_bg_A]
        loss_mapA = F.relu(self.conf.background_m - (desp_pts_fg_A - desp_pts_bg_A).norm(p=2, dim=1)) \
            .pow(2)
        desp_pts_fg_A2 = self.desp_A2[self.s_fg_A2]
        desp_pts_bg_A2 = self.desp_A2[self.s_bg_A2]
        loss_mapA2 = F.relu(self.conf.background_m - (desp_pts_fg_A2 - desp_pts_bg_A2).norm(p=2, dim=1)) \
            .pow(2)
        if not self.conf.A_A2_only:
            desp_pts_fg_B = self.desp_B[self.s_fg_B]
            desp_pts_bg_B = self.desp_B[self.s_bg_B]
            loss_mapB = F.relu(self.conf.background_m - (desp_pts_fg_B - desp_pts_bg_B).norm(p=2, dim=1)) \
                .pow(2)
            loss_map = torch.cat((loss_mapA, loss_mapA2, loss_mapB), dim=0)
        else:
            loss_map = torch.cat((loss_mapA, loss_mapA2), dim=0)

        # if self.conf.contrastive_non_zero:
        #     loss_map = loss_map[loss_map > 0]
        if loss_map.size(0) == 0:
            return self.np2torch([0.0])
        return loss_map

    def _background_same(self):
        desp_pts_bg_A = self.desp_A[self.s_bg_A_same]
        desp_pts_bg_A2 = self.desp_A2[self.s_bg_A2_same]
        if not self.conf.A_A2_only:
            desp_pts_bg_B = self.desp_B[self.s_bg_B_same]
            loss_map_AA2 = F.relu(
                (desp_pts_bg_A - desp_pts_bg_A2).norm(p=2, dim=1) - self.conf.background_consistent_m).pow(2)
            loss_map_AB = F.relu(
                (desp_pts_bg_A - desp_pts_bg_B).norm(p=2, dim=1) - self.conf.background_consistent_m).pow(
                2)
            loss_map_A2B = F.relu(
                (desp_pts_bg_A2 - desp_pts_bg_B).norm(p=2, dim=1) - self.conf.background_consistent_m).pow(2)
            loss_map = torch.cat((loss_map_AA2, loss_map_AB, loss_map_A2B), dim=0)
        else:
            loss_map = F.relu(
                (desp_pts_bg_A - desp_pts_bg_A2).norm(p=2, dim=1) - self.conf.background_consistent_m).pow(2)

        # if self.conf.contrastive_non_zero:
        #     loss_map = loss_map[loss_map > 0]
        if loss_map.size(0) == 0:
            return self.np2torch([0.0])
        return loss_map

    # # Experimental
    # def _contrastive_closest_non_matchAA2(self):  # Use A A2, Hard Negative on descriptor values
    #     # All dimension is [Nxx,3]
    #     m_pts_A = self.desp_A[self.s_match_hn_A]  # [Nam,3]
    #     m_pts_A2 = self.desp_A2[self.s_match_hn_A2]  # [Na2m,3] , Nam=Na2m
    #
    #     # All non-match points ~= mask_X_1d
    #     nm_pts_A = self.desp_A[self.non_match_idx_A]  # [Na,3]
    #     nm_pts_A2 = self.desp_A2[self.non_match_idx_A2]  # [Na2,3]
    #
    #     """
    #     Find the close descriptor points of A[match] vs. A2[non_match]
    #     """
    #     # expand m_pts_A to [Na2,Nam,3]
    #     match_pts_A_exp = m_pts_A.expand(nm_pts_A2.size(0), -1, -1)  # [Na2,Nam,3]
    #     obj_pts_A2_exp = nm_pts_A2.view(nm_pts_A2.size(0), 1, nm_pts_A2.size(1))  # [Na2,1,3]
    #     norm = (match_pts_A_exp - obj_pts_A2_exp).norm(p=2, dim=2)  # [Na2,Nam]
    #     # Select the hard negative
    #     select_negative = norm[norm < self.conf.neg_closest_AA2]
    #     if self.conf.neg_semi_hard_AA2:
    #         select_negative = norm[norm >= self.conf.match_m]
    #     loss_AA2 = F.relu(self.conf.neg_closest_AA2 - select_negative).pow(2)
    #     # loss_AA2 = loss_AA2[loss_AA2 > 0]
    #     """
    #     Find the close descriptor points of A2[match] vs. A[non_match]
    #     """
    #     # expand m_pts_A2 to [Na,Na2m,3]
    #     match_pts_A2_exp = m_pts_A2.expand(nm_pts_A.size(0), -1, -1)  # [Na,Na2m,3]
    #     obj_pts_A_exp = nm_pts_A.view(nm_pts_A.size(0), 1, nm_pts_A.size(1))  # [Na,1,3]
    #     norm2 = (match_pts_A2_exp - obj_pts_A_exp).norm(p=2, dim=2)  # [Na,Na2m]
    #     # Select the hard negative
    #     select_negative2 = norm2[norm2 < self.conf.neg_closest_AA2]
    #     if self.conf.neg_semi_hard_AA2:
    #         select_negative2 = norm2[norm2 >= self.conf.match_m]
    #     loss_A2A = F.relu(self.conf.neg_closest_AA2 - select_negative2).pow(2)
    #     # loss_A2A = loss_A2A[loss_A2A > 0]
    #
    #     loss_map = torch.cat((loss_AA2, loss_A2A), dim=0)
    #     if self.conf.contrastive_non_zero:
    #         loss_map = loss_map[loss_map > 0]
    #     if loss_map.size(0) == 0:
    #         return self.np2torch([0.0])
    #     return loss_map
    def _contrastive_closest_non_matchAA2(self):
        return torch.cat((self._contrastive_closest_non_matchA_A2(), self._contrastive_closest_non_matchA2_A()), dim=0)

    def _contrastive_closest_non_matchA_A2(self):  # Use A and A2, Hard Negative on descriptor values
        # Index expand
        non_match_idx_exp = self.s_mask_A2_hn_AA2.expand(self.s_match_hn_A.size(0), -1)  # [Nam, Nanm]
        match_A_idx_exp = self.s_match_hn_A.view(self.s_match_hn_A.size(0), 1)  # [Nam, 1]
        match_A2_idx_exp = self.s_match_hn_A2.view(self.s_match_hn_A2.size(0), 1)  # [Nam, 1]

        """
        Find the pixel within neg_closest_pixel distance between non-match point and match point in A2
        """
        inner_idx = (torch.abs(
            non_match_idx_exp % self.img_width - match_A2_idx_exp % self.img_width) < self.conf.neg_closest_pixel) * \
                    (torch.abs(
                        non_match_idx_exp / self.img_width - match_A2_idx_exp / self.img_width) < self.conf.neg_closest_pixel)
        inner_idx = inner_idx.to(self.device)

        """
        Find the close descriptor points of A[match] vs. A2[non_match]
        """
        m_pts_A = self.desp_A[match_A_idx_exp]
        nm_pts_A2 = self.desp_A2[non_match_idx_exp]
        norm = (nm_pts_A2 - m_pts_A).norm(p=2, dim=2)  # [Nam, Nanm]
        if self.conf.hard_negative_only_self:  # 's_match_hn_label', 's_mask_A_hn_AA2_cls', 's_mask_A2_hn_AA2_cls'
            cls_A2 = self.s_mask_A2_hn_AA2_cls.view(1, -1)  #
            cls_A = self.s_match_hn_label.view(-1, 1)
            cls_same = (cls_A - cls_A2) == 0  # [Nam,Nanm]

        hard_negative_pts = norm < self.conf.neg_closest_AA2
        if self.conf.hard_negative_only_self:
            select_negative = norm[~inner_idx * hard_negative_pts * cls_same]
        else:
            select_negative = norm[~inner_idx * hard_negative_pts]
        loss_AA2 = F.relu(self.conf.neg_closest_AA2 - select_negative).pow(2)
        loss_AA2 = loss_AA2[loss_AA2 > 0]

        loss_map = loss_AA2
        if loss_map.size(0) == 0:
            return self.np2torch([0.0])
        return loss_map

    def _contrastive_closest_non_matchA2_A(self):  # Use A2 and A, Hard Negative on descriptor values
        # Index expand
        non_match_idx_exp = self.s_mask_A_hn_AA2.expand(self.s_match_hn_A2.size(0), -1)  # [Nam, Nanm]
        match_A_idx_exp = self.s_match_hn_A.view(self.s_match_hn_A.size(0), 1)  # [Nam, 1]
        match_A2_idx_exp = self.s_match_hn_A2.view(self.s_match_hn_A2.size(0), 1)  # [Nam, 1]

        """
        Find the pixel within neg_closest_pixel distance between non-match point and match point in A2
        """
        inner_idx = (torch.abs(
            non_match_idx_exp % self.img_width - match_A_idx_exp % self.img_width) < self.conf.neg_closest_pixel) * \
                    (torch.abs(
                        non_match_idx_exp / self.img_width - match_A_idx_exp / self.img_width) < self.conf.neg_closest_pixel)
        inner_idx = inner_idx.to(self.device)

        """
        Find the close descriptor points of A[match] vs. A2[non_match]
        """
        m_pts_A2 = self.desp_A2[match_A2_idx_exp]
        nm_pts_A = self.desp_A[non_match_idx_exp]
        norm = (nm_pts_A - m_pts_A2).norm(p=2, dim=2)

        if self.conf.hard_negative_only_self:  # 's_match_hn_label', 's_mask_A_hn_AA2_cls', 's_mask_A2_hn_AA2_cls'
            cls_A = self.s_mask_A_hn_AA2.view(1, -1)  #
            cls_A2 = self.s_match_hn_label.view(-1, 1)
            cls_same = (cls_A - cls_A2) == 0  # [Nam,Nanm]

        hard_negative_pts = norm < self.conf.neg_closest_AA2
        if self.conf.hard_negative_only_self:
            select_negative = norm[~inner_idx * hard_negative_pts * cls_same]
        else:
            select_negative = norm[~inner_idx * hard_negative_pts]

        loss_A2A = F.relu(self.conf.neg_closest_AA2 - select_negative).pow(2)
        loss_A2A = loss_A2A[loss_A2A > 0]

        loss_map = loss_A2A
        if loss_map.size(0) == 0:
            return self.np2torch([0.0])
        return loss_map

    def _contrastive_closest_non_matchAB(self):  # Use A A2, Hard Negative on descriptor values
        # All dimension is [Nxx,3]
        if self.conf.hard_negative_only_self:
            return self.np2torch([0.0])
        pts_A = self.desp_A[self.s_mask_A_hn]  # [Na,3]
        pts_A2 = self.desp_A2[self.s_mask_A2_hn]  # [Na2,3]
        pts_B = self.desp_B[self.mask_B_1d]  # Total NB Points

        """
        Find the close descriptor points of A vs. B and A2 vs B
        """
        pts_A_exp = pts_A.expand(pts_B.size(0), -1, -1)
        pts_A2_exp = pts_A2.expand(pts_B.size(0), -1, -1)
        pts_B_exp = pts_B.view(pts_B.size(0), 1, pts_B.size(1))
        normAB = (pts_A_exp - pts_B_exp).norm(p=2, dim=2)
        normA2B = (pts_A2_exp - pts_B_exp).norm(p=2, dim=2)
        # Select the hard negative
        select_negativeAB = normAB[normAB < self.conf.neg_closest_AB]
        if self.conf.neg_semi_hard_AB:
            select_negativeAB = normAB[normAB >= self.conf.match_m]
        select_negativeA2B = normA2B[normA2B < self.conf.neg_closest_AB]
        if self.conf.neg_semi_hard_AB:
            select_negativeA2B = normA2B[normA2B >= self.conf.match_m]
        loss_AB = F.relu(self.conf.neg_closest_AB - select_negativeAB).pow(2)
        # loss_AB = loss_AB[loss_AB > 0]

        loss_A2B = F.relu(self.conf.neg_closest_AB - select_negativeA2B).pow(2)
        # loss_A2B = loss_A2B[loss_A2B > 0]

        loss_map = torch.cat((loss_AB, loss_A2B), dim=0)
        # if self.conf.contrastive_non_zero:
        loss_map = loss_map[loss_map > 0]
        if loss_map.size(0) == 0:
            return self.np2torch([0.0])
        return loss_map

    def get_match_distance(self):
        dist_map = self.tensor_mean(self._match_closest_pixel_distance().float())
        self.add_loss_record(['pixel_dist_avg'], [dist_map])
        return dist_map

    # Closest point error
    def _match_closest_pixel_distance(self):
        m_pts_A = self.desp_A[self.s_match_pix_A]  # [Nam,3]
        m_pts_A2 = self.desp_A2[self.s_match_pix_A2]  # [Na2m,3] , Nam=Na2m

        # All points [We can thus map to to the ground truth indices]
        nm_pts_A = self.desp_A  # [Na,3]
        nm_pts_A2 = self.desp_A2  # [Na2,3]

        match_pts_A_exp = m_pts_A.expand(nm_pts_A2.size(0), -1, -1)  # [Na2,Nam,3]
        obj_pts_A2_exp = nm_pts_A2.view(nm_pts_A2.size(0), 1, nm_pts_A2.size(1))  # [Na2,1,3]
        norm = (match_pts_A_exp - obj_pts_A2_exp).norm(p=2, dim=2)  # [Na2,Nam]
        a_match_a2_idx = norm.argmin(dim=0)  # <=> s_match_pix_A2 (Ground Truth)
        dist_mapAA2 = indices_1d_to_2d_distance_map(a_match_a2_idx, self.s_match_pix_A2, self.img_width)

        match_pts_A2_exp = m_pts_A2.expand(nm_pts_A.size(0), -1, -1)  # [Na,Na2m,3]
        obj_pts_A_exp = nm_pts_A.view(nm_pts_A.size(0), 1, nm_pts_A.size(1))  # [Na,1,3]
        norm2 = (match_pts_A2_exp - obj_pts_A_exp).norm(p=2, dim=2)  # [Na,Na2m]
        a2_match_a_idx = norm2.argmin(dim=0)  # <=> s_match_pix_A (Ground Truth)
        dist_mapA2A = indices_1d_to_2d_distance_map(a2_match_a_idx, self.s_match_pix_A, self.img_width)

        dist_map = torch.cat((dist_mapAA2, dist_mapA2A), dim=0)
        return dist_map
        # Closest point error

    def _match_closest_class_pixel_distance(self, net, use_fp16=False):
        m_pts_A = self.desp_A[self.s_match_pix_A]  # [Nam,3]
        m_pts_A2 = self.desp_A2[self.s_match_pix_A2]  # [Na2m,3] , Nam=Na2m
        Nam, Na2m = self.s_match_pix_A.size(0), self.s_match_pix_A2.size(0)
        all_label_A = self.clf_A  # , net, use_fp16)
        all_label_A2 = self.clf_A2  # , net, use_fp16)

        label_pts_A = all_label_A[self.s_match_pix_A]
        label_pts_A2 = all_label_A2[self.s_match_pix_A2]

        nm_pts_A = self.desp_A  # [Na,3]
        nm_pts_A2 = self.desp_A2  # [Na2,3]

        match_pts_A_exp = m_pts_A.expand(nm_pts_A2.size(0), -1, -1)  # [Na2,Nam,3]
        obj_pts_A2_exp = nm_pts_A2.view(nm_pts_A2.size(0), 1, nm_pts_A2.size(1))  # [Na2,1,3]
        norm = (match_pts_A_exp - obj_pts_A2_exp).norm(p=2, dim=2)  # [Na2,Nam]
        norm, idx_sort = norm.sort(dim=0)  # [Na2,Nam]
        a_match_a2_idx = torch.zeros(Nam).to(self.device)
        a_match_a2_idx_class = torch.zeros(Nam).to(self.device)
        for i in range(Nam):  # 100-300
            label_pt = label_pts_A[i].item()
            label_select = all_label_A2[idx_sort[:, i]].view(-1)  # Na2
            idx_arr = torch.nonzero(label_pt == label_select).view(-1)
            if idx_arr.size(0) > 0:
                k = idx_arr[0]
                if norm[k, i] < 3 and k < 10:
                    a_match_a2_idx_class[i] = idx_sort[k, i]
                else:
                    a_match_a2_idx_class[i] = idx_sort[0, i]
            else:
                a_match_a2_idx_class[i] = idx_sort[0, i]
            a_match_a2_idx[i] = idx_sort[0, i]
            # set_trace()
        dist_mapAA2 = indices_1d_to_2d_distance_map(a_match_a2_idx, self.s_match_pix_A2, self.img_width)
        dist_mapAA2_class = indices_1d_to_2d_distance_map(a_match_a2_idx_class, self.s_match_pix_A2, self.img_width)

        # del norm
        # gc.collect()
        # torch.cuda.empty_cache()

        match_pts_A2_exp = m_pts_A2.expand(nm_pts_A.size(0), -1, -1)  # [Na,Na2m,3]
        obj_pts_A_exp = nm_pts_A.view(nm_pts_A.size(0), 1, nm_pts_A.size(1))  # [Na,1,3]
        norm2 = (match_pts_A2_exp - obj_pts_A_exp).norm(p=2, dim=2)  # [Na,Na2m]
        norm2, idx_sort2 = norm2.sort(dim=0)  # [Na2,Nam]

        a2_match_a_idx = torch.zeros(Na2m).to(self.device)
        a2_match_a_idx_class = torch.zeros(Na2m).to(self.device)
        for i in range(Na2m):  # 100-300
            label_pt = label_pts_A2[i].item()
            label_select = all_label_A[idx_sort2[:, i]].view(-1)  # Na2
            idx_arr = torch.nonzero(label_pt == label_select).view(-1)
            if idx_arr.size(0) > 0:
                k = idx_arr[0]
                if norm2[k, i] < 3 and k < 10:
                    a2_match_a_idx_class[i] = idx_sort2[k, i]
                else:
                    a2_match_a_idx_class[i] = idx_sort2[0, i]
            else:
                a2_match_a_idx_class[i] = idx_sort2[0, i]
            a2_match_a_idx[i] = idx_sort2[0, i]
        dist_mapA2A = indices_1d_to_2d_distance_map(a2_match_a_idx, self.s_match_pix_A, self.img_width)
        dist_mapA2A_class = indices_1d_to_2d_distance_map(a2_match_a_idx_class, self.s_match_pix_A, self.img_width)
        print('AA2', dist_mapAA2.mean() - dist_mapAA2_class.mean(), ' A2A',
              dist_mapA2A.mean() - dist_mapA2A_class.mean())

        dist_map = torch.cat((dist_mapAA2, dist_mapA2A), dim=0)
        dist_map_class = torch.cat((dist_mapAA2_class, dist_mapA2A_class), dim=0)
        return dist_map, dist_map_class

    # def get_clf_label(self, desp, net, use_fp16=False):
    #     if use_fp16:
    #         pred_ = net.cls_forward(desp.half()).float()
    #     else:
    #         pred_ = net.cls_forward(desp, dim=0)
    #
    #     _, pred_label = torch.max(pred_.detach(), 1)
    #     return pred_label
    def get_all_clf_label(self):
        self.clf_A = self.get_clf_label(self.desp_A)
        self.clf_A2 = self.get_clf_label(self.desp_A2)
        if not self.conf.A_A2_only:
            self.clf_B = self.get_clf_label(self.desp_B)
        pass

    def get_clf_label(self, desp):
        # desp: NP,Dim
        with torch.no_grad():
            centers = self.centers.get_centers()  # NC,D
            desp_exp = desp.expand(centers.size(0), -1, -1)  # [NC,Np,D]
            centers_exp = centers.view(centers.size(0), 1, centers.size(1))  # [NC,1,D]
            norm = (desp_exp - centers_exp).norm(p=2, dim=2)  # [NC,Np]
            min_dist, ctr_label = norm.min(dim=0)
            ctr_label += 1  # Background=0, other class remap from 1
            ctr_label[min_dist > self.conf.background_m] = 0

        return ctr_label

    """
    Pre-allocated Center Loss
    """

    def get_triplet_center_loss(self):
        """
        :param net_center: Parameter from net, should detach before use
        :return:
        """
        # Detach -> requires_grad=False,but share storage
        if self.conf.triplet_sum_loss:
            triplet_center = self._triplet_center_loss().sum()
        else:
            triplet_center = self.tensor_mean(self._triplet_center_loss().float())
        self.add_loss_record(
            ['triplet_center'],
            [triplet_center])
        total_loss = triplet_center
        return total_loss

    def get_loss_centers(self):
        """
        :param net_center: Parameter from net, N_cls,N_dim
        :return:
        """
        center_loss = 0
        net_center = self.centers
        # # TODO: Balanced Sampling
        for cls in self.match_class_set:
            cls = cls.item()
            points_A = self.desp_A[self.mask_A_1d[self.fg_A_cls == cls]].detach()
            points_A2 = self.desp_A2[self.mask_A2_1d[self.fg_A2_cls == cls]].detach()
            points = torch.cat((points_A, points_A2), dim=0)
            center_loss += self.tensor_mean((points - net_center.get_center(cls)).norm(p=2, dim=1).pow(2))
        if not self.conf.A_A2_only:
            for cls in self.non_match_class_set:
                cls = cls.item()
                points = self.desp_B[self.mask_B_1d[self.fg_B_cls == cls]].detach()
                center_loss += self.tensor_mean((points - net_center.get_center(cls)).norm(p=2, dim=1).pow(2))
        # center_loss = torch.cat(center_loss).sum()  # TODO:Check

        # for cls in self.match_class_set:
        #     cls = cls.item()
        #     points_A = self.desp_A[self.mask_A_1d[self.fg_A_cls == cls]].detach()
        #     points_A2 = self.desp_A2[self.mask_A2_1d[self.fg_A2_cls == cls]].detach()
        #     points = torch.cat((points_A, points_A2), dim=0)
        #     center_loss.append((points - net_center.get_center(cls)).norm(p=1, dim=1))
        # if not self.conf.A_A2_only:
        #     for cls in self.non_match_class_set:
        #         cls = cls.item()
        #         points = self.desp_B[self.mask_B_1d[self.fg_B_cls == cls]].detach()
        #         center_loss.append((points - net_center.get_center(cls)).norm(p=1, dim=1))
        # center_loss = self.tensor_mean(torch.cat(center_loss, dim=0))  # TODO:Check

        self.add_loss_record(
            ['centers_loss'],
            [center_loss])
        return center_loss

    def _triplet_center_loss(self):
        # Construct Triplet Pairs
        # Anchor:    Mask A/A2 (sample?)
        # Positive:  Center A
        # Negative:  Centers not A (k)
        # Type:      APN
        centers = self.centers
        if len(centers) == 1:  # single object mode
            return self.np2torch([0.0])
        loss_map_A = []
        loss_map_A2 = []
        for cls in self.match_class_set:
            centers_no_self = []
            cls = cls.item()
            for k in centers.center_cls:  # Ignore 0, which is background
                if k != cls:
                    centers_no_self.append(centers[k])
            centers_no_self = torch.stack(centers_no_self)
            center = centers[cls][None]

            anchorsA = self.mask_A_1d[self.fg_A_cls == cls]
            anchorsA2 = self.mask_A2_1d[self.fg_A2_cls == cls]
            anc, pos, neg = construct_triplet_indices_APN(anchorsA, self.np2torch([0], torch.long),
                                                          self.np2torch(torch.arange(centers_no_self.size(0)),
                                                                        torch.long))

            loss_map_A.append(self.calculate_triplet_loss_map(self.desp_A[anc], center[pos], centers_no_self[neg],
                                                              self.conf.center_dist_diff))

            anc, pos, neg = construct_triplet_indices_APN(anchorsA2, self.np2torch([0], torch.long),
                                                          self.np2torch(torch.arange(centers_no_self.size(0)),
                                                                        torch.long))

            loss_map_A2.append(self.calculate_triplet_loss_map(self.desp_A2[anc], center[pos], centers_no_self[neg],
                                                               self.conf.center_dist_diff))
        loss_mapA = torch.cat(loss_map_A, dim=0)
        loss_mapA2 = torch.cat(loss_map_A2, dim=0)
        if not self.conf.A_A2_only:
            loss_map_B = []
            for cls in self.non_match_class_set:
                centers_no_self = []
                cls = cls.item()
                for k in centers.center_cls:  # Ignore 0, which is background
                    if k != cls:
                        centers_no_self.append(centers[k])
                centers_no_self = torch.stack(centers_no_self)
                center = centers[cls][None]

                anchorsB = self.mask_B_1d[self.fg_B_cls == cls]
                anc, pos, neg = construct_triplet_indices_APN(anchorsB, self.np2torch([0], torch.long),
                                                              self.np2torch(torch.arange(centers_no_self.size(0)),
                                                                            torch.long))

                loss_map_B.append(self.calculate_triplet_loss_map(self.desp_B[anc], center[pos], centers_no_self[neg],
                                                                  self.conf.center_dist_diff))
            loss_mapB = torch.cat(loss_map_B, dim=0)
            loss_map = torch.cat((loss_mapA, loss_mapA2, loss_mapB), dim=0)
        else:
            loss_map = torch.cat((loss_mapA, loss_mapA2), dim=0)

        if loss_map.size(0) == 0:
            return self.np2torch([0.0])
        return loss_map

    def get_space_normalization_loss(self):
        norm_A = F.relu(self.desp_A.norm(p=2, dim=1) - self.conf.norm_space_max).mean()
        norm_A2 = F.relu(self.desp_A2.norm(p=2, dim=1) - self.conf.norm_space_max).mean()
        if not self.conf.A_A2_only:
            norm_B = F.relu(self.desp_B.norm(p=2, dim=1) - self.conf.norm_space_max).mean()
            loss_norm = norm_A + norm_A2 + norm_B
        else:
            loss_norm = norm_A + norm_A2
        self.add_loss_record(['loss_norm'], [loss_norm])
        return loss_norm

    """
    Triplet Losses
    
    AP_N: Anchor-Positive set, Negative set (Grid of Nap*Nn)
    APN : Anchor set, Positive set, Negative set (Grid of Na*Np*Nn)
    APN1: Anchor-Positive-Negative triplets (Grid generation is not required)
    """

    def calculate_triplet_loss_map(self, anchor_pts, positive_pts, negative_pts, alpha):
        assert anchor_pts.size(0) == positive_pts.size(0) == negative_pts.size(0)
        if self.conf.triplet_square_norm:
            loss_map = F.relu(
                (anchor_pts - positive_pts).norm(p=2, dim=1).pow(2) -
                (anchor_pts - negative_pts).norm(p=2, dim=1).pow(2) + alpha)
        else:
            loss_map = F.relu(
                (anchor_pts - positive_pts).norm(p=2, dim=1) - (anchor_pts - negative_pts).norm(p=2, dim=1) + alpha)
        if self.conf.triplet_pow2:
            loss_map = loss_map.pow(2)
        if self.conf.triplet_non_zero:
            loss_map = loss_map[loss_map > 0]
        if loss_map.size(0) == 0:
            return self.np2torch([0.0])
        return loss_map

    def calculate_triplet_loss_map_from_norm(self, anc_pos_norm, anc_neg_norm, alpha):
        assert anc_pos_norm.size(0) == anc_neg_norm.size(0)
        if self.conf.triplet_square_norm:
            loss_map = F.relu(anc_pos_norm.pow(2) - anc_neg_norm.pow(2) + alpha)
        else:
            loss_map = F.relu(anc_pos_norm - anc_neg_norm + alpha)
        if self.conf.triplet_pow2:
            loss_map = loss_map.pow(2)
        if self.conf.triplet_non_zero:
            loss_map = loss_map[loss_map > 0]
        if loss_map.size(0) == 0:
            return self.np2torch([0.0])
        return loss_map

    def np2torch(self, data, dtype=torch.float):
        if isinstance(data, torch._C._TensorBase):
            return data.clone().to(dtype).to(self.device)
        else:
            return torch.tensor(data).to(dtype).to(self.device)

    def tensor_mean(self, data):
        if data.size(0) == 0:
            return torch.tensor(0).to(torch.float).to(self.device)
        else:
            return data.mean()

    # Original Version
    def get_discriminative_loss(self, net, use_fp16=False):
        # Extract sample points
        # s_cls_A, s_cls_A2, s_cls_B, s_cls_A_bk
        # A_class, B_class = self.objA_class.item(), self.objB_class.item()

        A_points = self.desp_A[self.s_cls_A]
        A2_points = self.desp_A2[self.s_cls_A2]
        if not self.conf.A_A2_only:
            B_points = self.desp_B[self.s_cls_B]

            A_bk_points = self.desp_A[self.s_cls_bk_A]

            target_label = self.s_cls_labels.to(torch.long).to(self.device)
            points = torch.cat((A_points, A2_points, A_bk_points, B_points), dim=0)
            if use_fp16:
                pred_ = net.cls_forward(points.half()).float()
            else:
                pred_ = net.cls_forward(points)
            # pred_label = self.get_clf_label(points)
        else:
            A_bk_points = self.desp_A[self.s_cls_bk_A]
            target_label = self.s_cls_labels[0:A_points.size(0) + A2_points.size(0) + A_bk_points.size(0)].to(
                torch.long).to(self.device)
            points = torch.cat((A_points, A2_points, A_bk_points), dim=0)
            if use_fp16:
                pred_ = net.cls_forward(points.half()).float()
            else:
                pred_ = net.cls_forward(points)
            # pred_label = self.get_clf_label(points)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_, target_label)

        _, pred_label = torch.max(pred_.detach(), 1)

        total = len(pred_label)
        correct = (pred_label == target_label).sum().to(torch.float)
        accuracy = correct / total

        self.add_loss_record(['discriminative', 'class_accuracy'], [loss, accuracy])
        return loss

    # Metric Version -> Hard Triplet Mining
    def get_hard_metric_loss(self, epoch=0):
        """
        Mask: mask_A_1d, mask_A2_1d, mask_B_1d
        Class label: fg_A_cls, fg_A2_cls, fg_B_cls
        Predicted Class label: clf_A, clf_A2, clf_B
        """
        # Find Misclassified Points
        mis_A_fg = (self.clf_A[self.mask_A_1d] != self.fg_A_cls).nonzero().view(-1)
        mis_A_bg = (self.clf_A[self.bg_A_1d] != 0).nonzero().view(-1)
        fg_A_pts = self.desp_A[self.mask_A_1d][mis_A_fg]
        bg_A_pts = self.desp_A[self.bg_A_1d][mis_A_bg]
        fg_A_cls = self.fg_A_cls[mis_A_fg]

        mis_A2_fg = (self.clf_A2[self.mask_A2_1d] != self.fg_A2_cls).nonzero().view(-1)
        mis_A2_bg = (self.clf_A2[self.bg_A2_1d] != 0).nonzero().view(-1)
        fg_A2_pts = self.desp_A2[self.mask_A2_1d][mis_A2_fg]
        bg_A2_pts = self.desp_A2[self.bg_A2_1d][mis_A2_bg]
        fg_A2_cls = self.fg_A2_cls[mis_A2_fg]

        if not self.conf.A_A2_only:
            mis_B_fg = (self.clf_B[self.mask_B_1d] != self.fg_B_cls).nonzero().view(-1)
            mis_B_bg = (self.clf_B[self.bg_B_1d] != 0).nonzero().view(-1)
            fg_B_pts = self.desp_B[self.mask_B_1d][mis_B_fg]
            bg_B_pts = self.desp_B[self.bg_B_1d][mis_B_bg]
            fg_B_cls = self.fg_B_cls[mis_B_fg]
            fg_points = torch.cat((fg_A_pts, fg_A2_pts, fg_B_pts), dim=0)
            fg_cls_label = torch.cat((fg_A_cls, fg_A2_cls, fg_B_cls))
            bg_points = torch.cat((bg_A_pts, bg_A2_pts, bg_B_pts), dim=0)
        else:
            fg_points = torch.cat((fg_A_pts, fg_A2_pts), dim=0)
            fg_cls_label = torch.cat((fg_A_cls, fg_A2_cls))
            bg_points = torch.cat((bg_A_pts, bg_A2_pts), dim=0)  # Nbg,ND
        # # TODO: Sampling here
        max_bg_points = min(self.conf.hm_bg_points_start + self.conf.hm_bg_increase_epoch * epoch,
                            self.conf.hm_bg_points_max)
        max_fg_points = min(self.conf.hm_fg_points_start + self.conf.hm_fg_increase_epoch * epoch,
                            self.conf.hm_fg_points_max)

        sample_bg_n = min(bg_points.size(0), max_bg_points)
        bg_points = bg_points[torch.randperm(bg_points.size(0))[0:sample_bg_n].to(self.device), :]
        sample_fg_n = min(fg_points.size(0), max_fg_points)
        fg_idx = torch.randperm(fg_points.size(0))[0:sample_fg_n].to(self.device)
        fg_points = fg_points[fg_idx, :]
        fg_cls_label = fg_cls_label[fg_idx]
        # Background contrastive for wrong points
        # 1. GT: Bg, but close to other cluster => Do bk same on them
        if bg_points.size(0) >= 2:
            half_size = bg_points.size(0) // 2
            loss_map_bg_same = F.relu((bg_points[0:half_size, :] - bg_points[half_size:2 * half_size, :]) \
                                      .norm(p=2, dim=1) - self.conf.background_consistent_m).pow(2)
        else:
            loss_map_bg_same = self.np2torch([0.0])

        # # bg points should be far from all class centers
        if bg_points.size(0) > 0:
            centers = self.centers.get_centers()  # Detached centers, Nc,ND
            centers_exp = centers.view(centers.size(0), 1, centers.size(1))  # [NC,1,D]
            bg_points_exp = bg_points.expand(centers.size(0), -1, -1)  # [NC,Nbg,D]
            norm = (bg_points_exp - centers_exp).norm(p=2, dim=2).view(-1)  # [NC*Np]
            loss_map_bg_centers = F.relu(self.conf.background_m - norm).pow(2)
        else:
            loss_map_bg_centers = self.np2torch([0.0])

        # 2. GT: Fg classes, but close to other cluster => Triplet loss for all wrong foreground points
        loss_map_triplet = []
        for cls in self.match_class_set:
            centers_no_self = []
            cls = cls.item()
            for k in self.centers.center_cls:  # Ignore 0, which is background
                if k != cls:
                    centers_no_self.append(self.centers[k])
            centers_no_self = torch.stack(centers_no_self)
            center = self.centers[cls][None]

            anchors = (fg_cls_label == cls).nonzero().view(-1)  # Index
            anc, pos, neg = construct_triplet_indices_APN(anchors, self.np2torch([0], torch.long),
                                                          self.np2torch(torch.arange(centers_no_self.size(0)),
                                                                        torch.long))

            loss_map_triplet.append(self.calculate_triplet_loss_map(fg_points[anc], center[pos], centers_no_self[neg],
                                                                    self.conf.center_dist_diff))
        loss_map_triplet = torch.cat(loss_map_triplet, dim=0)
        if loss_map_triplet.size(0) == 0:
            loss_map_triplet = self.np2torch([0.0])

        sum_triplet = loss_map_triplet.sum()
        sum_bg_centers = loss_map_bg_centers.sum()
        sum_bg_same = loss_map_bg_same.sum()
        self.add_loss_record(['loss_map_triplet', 'loss_map_bg_centers', 'loss_map_bg_same'],
                             [sum_triplet, sum_bg_centers, sum_bg_same])

        loss = sum_triplet + sum_bg_centers + sum_bg_same

        # loss_map = torch.cat((loss_map_triplet, loss_map_bg_same), dim=0)  # No bg center
        # print('loss_map_triplet', loss_map_triplet.sum(), ' loss_map_bg_centers', loss_map_bg_centers.sum(),
        #       ' loss_map_bg_same', loss_map_bg_same.sum())
        # set_trace()
        # Center Triplet for wrong points

        # Compatible code
        A_points = self.clf_A[self.s_cls_A]
        A2_points = self.clf_A2[self.s_cls_A2]
        if not self.conf.A_A2_only:
            B_points = self.clf_B[self.s_cls_B]
            A_bk_points = self.clf_A[self.s_cls_bk_A]
            target_label = self.s_cls_labels.to(torch.long).to(self.device)
            pred_label = torch.cat((A_points, A2_points, A_bk_points, B_points), dim=0)
        else:
            A_bk_points = self.clf_A[self.s_cls_bk_A]
            target_label = self.s_cls_labels[0:A_points.size(0) + A2_points.size(0) + A_bk_points.size(0)].to(
                torch.long).to(self.device)
            pred_label = torch.cat((A_points, A2_points, A_bk_points), dim=0)

        total = len(pred_label)
        correct = (pred_label == target_label).sum().to(torch.float)
        accuracy = correct / total
        self.add_loss_record(['discriminative', 'class_accuracy'], [loss, accuracy])
        return loss


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def indices_1d_to_2d_distance_map(src_1d, tgt_1d, width):
    src_1d, truth_1d = src_1d.clone().to(torch.long), tgt_1d.clone().to(torch.long)
    src_u, src_v = (src_1d % width, (src_1d / width))
    tgt_u, tgt_v = (tgt_1d % width, (tgt_1d / width))
    # TODO: Check sqrt
    dist_map = ((src_u - tgt_u).pow(2).to(torch.float) + (src_v - tgt_v).pow(2).to(torch.float)).sqrt()
    return dist_map


def construct_triplet_indices_APN(anchor, positive, negative):  # 70ms/1.62ms for 300,300,300 in CPU/GPU
    """
    Anchor set, Positive set, Negative set  (1D pixel indices)
    Any element in anchor set is pos./neg. with any element from pos./neg. set
    Return 1d indices for accessing descriptor
    """
    idx_a, idx_p, idx_n = torch.meshgrid((anchor, positive, negative))
    return idx_a.flatten(), idx_p.flatten(), idx_n.flatten()


def construct_triplet_indices_AP_N(anchor, positive, negative):  # 2.47ms/659us for 300,300,300 in CPU/GPU
    """
    Anchor-Positive set, Negative set  (1D pixel indices)
    Any pair in Anchor-Positive set is neg. with any element from neg. set
    """
    assert anchor.size(0) == positive.size(0)
    ap_indices = torch.arange(anchor.size(0))
    idx_ap, idx_n = torch.meshgrid((ap_indices, negative))
    idx_ap = idx_ap.flatten()
    return anchor[idx_ap], positive[idx_ap], idx_n.flatten()

import numpy as np
import os, sys, yaml

# from PIL import Image  # CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
from torchvision import transforms
from torch.utils.data import Dataset
import cv2  # For testing code only

#
# from ipdb import set_trace
"""
StaticDataset: For stable testing result, fixed random seed
A,A2 Only , No patch and multi patch
"""


class VisualDataset(Dataset):
    def __init__(self, yaml_dict):
        # Access all parameters directly
        self.__dict__.update(yaml_dict)
        transform_list = []
        transform_list.extend([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
        self.transform = transforms.Compose(transform_list)
        self.folders = self.id_to_scenes[self.use_id]
        self.rs = np.random.RandomState(os.getpid())

    def __getitem__(self, idx):  # Sampling Dataset
        # Sample scene A and object A(A2) id
        folder, folder_n = self.rs.permutation(self.folders)[0]

        # p_end = p_start + self.rs.randint(1, self.match_frame_near + 1)  # TODO: Fixed seed randomization here
        # # [img_m11,img_m12,img_m21,img_m22,...], 2D image

        match_ids = self.sample_pair_in_range(folder_n, self.frame_near)
        match_folder_ids = [(folder, match_ids)]
        match_img = [self.open_img(folder, 'color', p, 0)
                     for folder, pair in match_folder_ids for p in pair]
        img_A, img_A2 = match_img
        img_A = img_A[:, :, [2, 1, 0]]
        img_A2 = img_A2[:, :, [2, 1, 0]]
        img_A, img_A2 = tuple(map(self.transform, [img_A, img_A2]))
        return img_A, img_A2

    def open_img(self, folder, info_type, idx, mode):
        if mode == 0:  # RGB
            return cv2.imread(os.path.join(self.base_dir, folder, '{:06}-{}.png'.format(idx, info_type)))
        else:  # Mask,Depth
            return cv2.imread(os.path.join(self.base_dir, folder, '{:06}-{}.png'.format(idx, info_type)),
                              cv2.IMREAD_ANYDEPTH)

    def sample_pair_in_range(self, N, near_range):
        N = int(N)
        n1 = self.rs.randint(N)
        while True:
            n2 = self.rs.randint(n1 - near_range, n1 + near_range) % N
            if n2 != n1:
                break
        return [n1, n2]


def test():
    with open('./exps_iros/others/0124_8c_200.yaml', 'r') as stream:
        conf = yaml.load(stream)  # Full config for reference
    ds = VisualDataset(conf['visual_data'])


if __name__ == '__main__':
    test()

import torch
import torch.nn as nn
import torch.nn.functional as f


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = f.relu(self.bn1(self.conv1(x)))
        out = f.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = f.relu(out)
        return out


class Center(nn.Module):
    def __init__(self, dim, seed=None, learn=False):
        super(Center, self).__init__()
        self.dim = dim
        self._center_keys = []
        self.center_cls = []
        self.learn = learn
        if self.learn:
            print('Center Learnable')
        else:
            print('Center Fixed')
        if seed is not None:
            if isinstance(seed, torch._C._TensorBase):
                # 2D seed
                for i in range(seed.size(0)):
                    self.add_center(i, seed[i])
            elif isinstance(seed, dict):
                for k, v in seed.items():
                    self.add_center(k, v)
            else:
                raise ValueError

    def add_center(self, cls_n, v):
        k = 'center_' + str(cls_n)
        assert isinstance(v, torch._C._TensorBase) and v.size(0) == self.dim, 'Center Dimension Error'
        setattr(self, k, nn.Parameter(v, requires_grad=self.learn))
        print('Center {},{} Added'.format(k, getattr(self, k).data))
        self._center_keys.append(k)
        self.center_cls.append(cls_n)

    def print_centers(self):
        for k in self._center_keys:
            print(k, getattr(self, k).data)

    def __getitem__(self, cls_n):  # Not in graph
        return getattr(self, 'center_' + str(cls_n)).detach()

    def get_center(self, cls_n):  # For update center
        return getattr(self, 'center_' + str(cls_n))

    def get_centers(self):
        return torch.cat([getattr(self, k)[None].detach() for k in self._center_keys], dim=0)

    def __len__(self):
        return len(self.center_cls)

    def forward(self, *input):
        pass


class FPN(nn.Module):
    def __init__(self, block, num_blocks, desp_dim=3, class_dims=None, use_class=False, use_center=False,
                 center_seeds=None, center_class=None, learn_center=False, split=False, gpu_seq=None, split_desp=False,
                 s_feature_dim=None, s_class_dim=None, s_seg_class_only=False):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.use_class = use_class
        self.gpu_seq = gpu_seq
        if use_center:
            if center_seeds is not None:
                self.centers = Center(desp_dim, center_seeds, learn_center)  # requires_grad
            else:
                if split_desp:
                    print('Use splited descriptor space: ', s_feature_dim, ' , ', s_class_dim)
                    seed = torch.randn(len(center_class), s_class_dim)
                    self.centers = Center(s_class_dim, {k: seed[i] for i, k in enumerate(center_class)},
                                          learn_center)  # requires_grad
                else:
                    seed = torch.randn(len(center_class), desp_dim)
                    self.centers = Center(desp_dim, {k: seed[i] for i, k in enumerate(center_class)},
                                          learn_center)  # requires_grad
        if self.use_class:
            if s_seg_class_only and split_desp:
                print('Clf only on class dims, dim=', s_class_dim, ',', class_dims)
                self.clf = DescClf(s_class_dim, class_dims)
            else:
                self.clf = DescClf(desp_dim, class_dims)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.conv = nn.Conv2d(1024, desp_dim, kernel_size=1, stride=1)  # Descriptor  Dimension

        # self.dropout = nn.Dropout2d(p=0.2)

        # Top-down layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # # Smooth layers
        # self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upsample_120_160 = lambda x: nn.functional.interpolate(x, size=(120, 160), mode='bilinear',
                                                                    align_corners=True)
        self.upsample_480_640 = lambda x: nn.functional.interpolate(x, size=(480, 640), mode='bilinear',
                                                                    align_corners=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def to_gpus(self):
        gpu0_names = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'conv', 'toplayer', 'latlayer1',
                      'latlayer2', 'clf']
        gpu1_names = ['centers']
        module_keys = self.__dict__['_modules'].keys()
        parameter_keys = self.__dict__['_parameters'].keys()
        for m in module_keys:
            if m in gpu0_names:
                setattr(self, m, getattr(self, m).to(self.gpu_seq[0]))
            elif m in gpu1_names:
                setattr(self, m, getattr(self, m).to(self.gpu_seq[1]))
            else:
                print('Ignore module ', m)
        for p in parameter_keys:
            if p in gpu0_names:
                setattr(self, p, nn.Parameter(getattr(self, p).to(self.gpu_seq[0])))
            elif p in gpu1_names:
                setattr(self, p, nn.Parameter(getattr(self, p).to(self.gpu_seq[1])))
            else:
                print('Ignore parameter ', p)
        print('Module is split into GPUs.')

    @staticmethod
    def _upsample_add(x, y):
        """
        Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, h, w = y.size()

        return f.interpolate(x, size=(h, w), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = f.relu(self.bn1(self.conv1(x)))
        c1 = f.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, c2)

        # Upsample-merge
        p2 = self.upsample_120_160(p2)
        p3 = self.upsample_120_160(p3)
        p4 = self.upsample_120_160(p4)
        p5 = self.upsample_120_160(p5)

        t = torch.cat((p2, p3, p4, p5), 1)

        t = self.conv(t)
        t = self.upsample_480_640(t).to(self.gpu_seq[1])
        return t

    def cls_forward(self, desp_points):
        desp_points = desp_points.to(self.gpu_seq[0])
        return self.clf(desp_points).to(self.gpu_seq[1])


def fpn50(model_conf, device):
    desp_dim = model_conf.desp_dim
    dims = model_conf.class_dims if model_conf.use_class else None
    # Two Initialization methods for centers
    # 1. Dictionary of center_cls:center values
    # 2. list of center_cls, randomly generated center values from normal distribution
    if model_conf.center_dict is not None:
        center_dict = model_conf.center_dict
        print('Use center dict initialization')
        center_dict = {k: torch.as_tensor(v).float() for k, v in center_dict.items()}
        class_list = list(center_dict.keys())
    else:
        center_dict = None
        class_list = model_conf.center_classes  # For center keys
        print('Use randomly generated centers')
    split = True if model_conf.split_model else False
    split_desp = True if model_conf.split_desp else False
    s_feature_dim = None if not split_desp else model_conf.s_feature_dim
    s_class_dim = None if not split_desp else model_conf.s_class_dim
    s_seg_class_only = False if not split_desp else model_conf.s_seg_class_only
    return FPN(Bottleneck, [3, 4, 6, 3], desp_dim, use_center=True, class_dims=dims, use_class=model_conf.use_class,
               center_seeds=center_dict, center_class=class_list, learn_center=model_conf.learn_center, split=split,
               gpu_seq=model_conf.gpu_seq, split_desp=split_desp, s_feature_dim=s_feature_dim, s_class_dim=s_class_dim,
               s_seg_class_only=s_seg_class_only)


class DescClf(nn.Module):
    def __init__(self, desp_dim, class_dims):
        super(DescClf, self).__init__()
        dims = [desp_dim]
        dims.extend(class_dims)  # desp_dim,*class_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=0.2))
        self.clf = nn.Sequential(*layers)

    def forward(self, desp):  # desp are sample points
        """
        :param desp: N,desp_dim
        :return: Logits, (N, class number)
        """
        return self.clf(desp)


def test():
    net = FPN(Bottleneck, [3, 4, 6, 3], 3)
    fms = net(torch.randn(1, 3, 480, 640))
    for fm in fms:
        print(fm.size())

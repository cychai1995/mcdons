import numpy as np
import time, math, os, errno
import torch
from torch.utils.data import Dataset
from torch import nn
import pickle, os
from torch.optim import lr_scheduler
import matplotlib
import prodict, yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.init as init


def load_conf(path):
    with open(path, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
    return prodict.Prodict.from_dict(yaml_dict)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def secondSince(since):
    now = time.time()
    s = now - since
    return s


def check_path(path):
    try:
        os.makedirs(path)  # Support multi-level
        print(path, ' created')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        # print(path, ' exists')


class TrainingProgress:
    def __init__(self, path, header, tp_step=None, data_key_list=None, data_dict=None, meta_dict=None, epoch_dict=None,
                 restore=False):
        """
        * Init Dict for storing key-value data
        * Configure Saving filename and Path
        * Restoring function

        Header => Filename header
        Data Dict => Appendable data (loss,time,acc....)
        Meta Dict => One time data (config,weight,.....)
        """
        self.filename = os.path.join(path, header)
        check_path(path)
        if restore:
            assert tp_step is not None, 'Explicitly assign the TP step you want to restore'
            self.restore_progress(tp_step)
        else:  # Two Initialization Methods for data_dict, keys or a dict, select one
            if data_key_list is not None:
                self.data_dict = {}
                for k in data_key_list:
                    self.data_dict[k] = []
            else:
                self.data_dict = {} if data_dict is None else data_dict  # record sequential data values
            self.meta_dict = {} if meta_dict is None else meta_dict
            # key=['train'/'test'/...]+str(epoch), val=dict of data values
            self.epoch_dict = {} if epoch_dict is None else epoch_dict

    def add_meta(self, new_dict):
        self.meta_dict.update(new_dict)

    def get_meta(self, key):
        try:
            return self.meta_dict[key]
        except KeyError:  # New key
            print('TP Error: Cannot find meta, key=', key)
            return None

    def record_data(self, new_dict, display=False):
        for k, v in new_dict.items():
            try:
                # if math.isnan(v):
                #     print('TP Warning: Ignore NaN value')
                # else:
                self.data_dict[k].append(v)
            except AttributeError:  # Append fail
                print('TP Error: Cannot Record data, key=', k)
            except KeyError:  # New key
                print('TP Warning: Add New Appendable data, key=', k)
                self.data_dict[k] = [v]
        if display:
            print('TP Record new data: ', new_dict)
        pass

    def record_epoch(self, epoch, prefix, new_dict, display=False):  # use this
        # record every epoch, prefix=train/test/validation....
        key = prefix + str(epoch)
        if key in self.epoch_dict.keys():
            # print('TP Warning: Epoch Data with key={} is overwritten'.format(key))
            self.epoch_dict[key].update(new_dict)
        else:
            self.epoch_dict[key] = new_dict
        if display:
            print(key, new_dict)

    def get_epoch_data(self, data_key, prefix, ep_start, ep_end, ep_step=1):
        data = []
        for ep in range(ep_start, ep_end, ep_step):
            key = prefix + str(ep)
            try:
                data.append(self.epoch_dict[key][data_key])
            except KeyError:
                print('TP Warning, Invalid epoch=', ep, ' Data Ignored!')
        return data

    def save_progress(self, tp_step, override_path=None):
        name = self.filename + str(tp_step) + '.tpdata' if override_path is None else override_path
        check_path(os.path.dirname(name))
        with open(name, "wb") as f:
            pickle.dump((self.data_dict, self.meta_dict, self.epoch_dict), f, protocol=2)

    def restore_progress(self, tp_step, override_path=None):
        name = self.filename + str(tp_step) + '.tpdata' if override_path is None else override_path
        with open(name, 'rb') as f:
            self.data_dict, self.meta_dict, self.epoch_dict = pickle.load(f)

    def plot_epoch_data(self, prefix, ep_start, ep_end, save_path, title, ep_step=1):  # [ep_start,ep_end]
        ep_end += 1
        data_keys = list(self.epoch_dict[prefix + str(ep_start)].keys())
        data_keys.sort()  # Item keys
        append_dict = {}
        for ep in range(ep_start, ep_end, ep_step):
            key = prefix + str(ep)
            for k, v in self.epoch_dict[key].items():
                try:
                    append_dict[k].append(v)
                except KeyError:
                    append_dict[k] = [v]
        n_cols = 3
        n_rows = int(len(data_keys) / n_cols + 1)
        fig = plt.figure(dpi=800, figsize=(n_cols * 3, n_rows * 3))
        fig.suptitle(title)
        x_ticks = list(range(ep_start, ep_end, ep_step))
        keys = sorted(append_dict.keys())
        # for i, (k, v) in enumerate(append_dict.items()):
        for i, k in enumerate(keys):
            v = append_dict[k]
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.grid(True)
            ax.plot(x_ticks, v)
            ax.set_xticks(x_ticks)
            ax.xaxis.set_tick_params(labelsize=4)
            ax.set_title(k)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        check_path(os.path.dirname(save_path))
        plt.savefig(save_path)

    def save_conf(self, dict):
        path = self.filename + 'conf.yaml'
        with open(path, 'w') as outfile:
            yaml.dump(dict, outfile)


class LearningRateScheduler:  # Include torch.optim.lr_scheduler
    def __init__(self, mode, param_groups, lr_rates=None, lr_epochs=None, lr_loss=None, lr_init=None,
                 lr_decay_func=None,
                 torch_lrs='ReduceLROnPlateau', torch_lrs_param={'mode': 'min', 'factor': 0.5, 'patience': 20}):
        self.mode = mode
        if isinstance(param_groups, torch.optim.Optimizer):
            Warning('Deprecated usage, pass list of param group instead')
            self.groups = param_groups.param_groups
        else:
            assert isinstance(param_groups, list)
            self.groups = param_groups  # the specific param group to be controlled
        self.rate = lr_init
        # Check each mode
        if self.mode == 'epoch':
            self.lr_rates = lr_rates  # only single value if decay mode else list of rate
            self.epoch_targets = lr_epochs
            assert (0 <= len(self.lr_rates) - len(self.epoch_targets) <= 1), "Learning rate scheduler setting error."
            self.rate_func = self.lr_rate_epoch
            self.adjust_learning_rate(self.rate)

        elif self.mode == 'loss':
            self.lr_rates = lr_rates
            self.loss_targets = lr_loss
            assert (0 <= len(self.lr_rates) - len(self.loss_targets) <= 1), 'Learning rate scheduler setting error.'
            self.rate_func = self.lr_rate_loss
            self.adjust_learning_rate(self.rate)

        elif self.mode == 'decay':
            self.lr_rates = lr_rates  # only single value if decay mode else list of rate
            self.decay_func = lr_decay_func
            self.rate_func = self.lr_rate_decay
            # raise NotImplementedError  # Zzz....

        elif self.mode == 'torch':
            raise NotImplementedError('TODO: Modify to based on param group')
            # Should set the lr scheduler name in torch.optim.scheduler
            assert torch_lrs_param is not None, "Learning rate scheduler setting error."

            if torch_lrs == 'ReduceLROnPlateau':
                self.torch_lrs = getattr(lr_scheduler, 'ReduceLROnPlateau')(self.optimizer,
                                                                            **torch_lrs_param)  # instance
            else:
                raise NotImplementedError
            self.rate_func = self.torch_lrs.step
        else:
            raise NotImplementedError("Learning rate scheduler setting error.")
        print('Learning rate scheduler: Mode=', self.mode, ' Learning rate=', self.rate)

    def step(self, param_dict, display=True):
        if self.mode == 'torch':
            self.rate_func(param_dict[self.mode])
        else:
            new_rate, self.next = self.rate_func(param_dict[self.mode])
            if new_rate == self.rate:
                return
            else:
                self.rate = new_rate
                if display:
                    print('Learning rate scheduler: Mode=', self.mode, ' New Learning rate=', new_rate,
                          ' Next ', self.mode, ' target=', self.next)
                self.adjust_learning_rate(self.rate)

    def lr_rate_epoch(self, epoch):
        for idx, e in enumerate(self.epoch_targets):
            if epoch < e:
                # next lr rate, next epoch target for changing lr rate
                return self.lr_rates[idx], self.epoch_targets[idx]
        return self.lr_rates[-1], -1  # Last(smallest) lr rate

    def lr_rate_loss(self, loss):
        for idx, l in enumerate(self.loss_targets):
            if loss > l:
                return self.lr_rates[idx], self.loss_targets[idx]  # next lr rate, next loss target for changing lr rate
        return self.lr_rates[-1], -1  # Last(smallest) lr rate

    def lr_rate_decay(self, n):
        rate = self.rate * self.decay_func(n)
        return rate, -1

    def adjust_learning_rate(self, lr):
        for group in self.groups:
            group['lr'] = lr


def initialize_weight(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            print('Conv2d Init')
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            print('BatchNorm Init')
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            print('Linear Init')
            nn.init.xavier_uniform(m.weight)
            nn.init.constant_(m.bias, 1e-3)


def partial_load_weight(dict_src, dict_tgt):
    """
    Example Usage
    >>> dict_src = src_net.state_dict()
    >>> dict_tgt = tgt_net.state_dict()
    >>> dict_tgt = partial_load_weight(dict_src, dict_tgt)
    >>> tgt_net.load_state_dict(dict_tgt)
    """

    keys_src = dict_src.keys()
    for k in dict_tgt.keys():
        if k in keys_src:
            if dict_tgt[k].data.size() == dict_src[k].data.size():
                dict_tgt[k].data = dict_src[k].data.clone()
                # print(k, ' Loaded')
            else:
                pass
                # print(k, ' Size Mismatched')
    return dict_tgt


class ValueMeter:
    def __init__(self):
        self.data_dict = {}

    def record_data(self, dict):
        # assume values are numpy array or python number
        for k, v in dict.items():
            try:
                self.data_dict[k].append(v)
            except KeyError:
                self.data_dict[k] = [v]
            if math.isnan(v):
                print('Warning: Nan in Loss Recorder, key=', k)

    def avg(self):
        result_dict = {}
        for k, v in self.data_dict.items():
            result_dict[k] = np.mean(v)
        return result_dict

    def reset(self):
        self.data_dict = {}


class ConfNamespace(object):
    def __init__(self, conf_dict, override_dict=None):
        self.__dict__.update(conf_dict)
        if override_dict is not None:
            valid_conf = {k: v for k, v in override_dict.items() if
                          (v is not None) and (v is not False)}
            # Argparse default False if action='store_true'
            self.__dict__.update(valid_conf)


# def initialize_weight2(net):
#     conv_2d = 0
#     bn_2d = 0
#     fc = 0
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#             conv_2d += 1
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)
#             bn_2d += 1
#         elif isinstance(m, nn.Linear):
#             nn.init.normal(m.weight, std=1e-3)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 1e-4)
#             fc += 1
#     print('Weight Initialized, {} Conv2D Layers, {} BatchNorm2D Layers, {} Linear layers'.format(conv_2d, bn_2d, fc))


# def initialize_weight_fin(net, mean=0, bias=1e-4):
#     for name, param in net.state_dict().items():
#         if name.find('weight') != -1:
#             size = param.size()  # returns a tuple
#             print('Init weight name',name,' size:', size)
#             # fan_out = size[0]  # number of rows
#             fan_in = size[1]  # number o
#             nn.init.normal(param, mean=mean, std=np.sqrt(1 / fan_in))
#         elif name.find('bias') != -1:
#             nn.init.constant(param, bias)
#     print('All Weight Initialized (Fan-in mode)')


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def train_valid_split(dataset, train_ratio, random_indices=None):
    N = len(dataset)
    train_n = int(train_ratio * N)
    valid_n = N - train_n
    assert train_ratio <= 1
    print('Training set:', train_n, ' , Validation set:', valid_n)
    indices = random_indices if random_indices is not None else np.random.permutation(N)
    assert len(indices) == N
    return Subset(dataset, indices=indices[0:train_n]), Subset(dataset, indices=indices[train_n:N])


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__ == '__main__':
    a = TrainingProgress(
        '/home/chai/projects/visual_descriptor_learning/progress/101701', 'progress', 20,
        restore=True)
    a.plot_epoch_data('train', 1, 20, 'aaa.png', 'Train Loss')

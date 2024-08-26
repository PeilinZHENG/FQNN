import os
import shutil
import math
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader, _utils
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from typing import Sized, Iterator


def Optimizer(opt, params, lr=1e-3, weight_decay=0., momentum=0., betas=(0.9, 0.999), max_iter=20):
    # params = filter(lambda p: p.requires_grad, net.parameters())
    if opt == 'Adadelta':
        return optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif opt == 'Adagrad':
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif opt == 'Adam':
        return optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == 'AdamW':
        return optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == 'SparseAdam':
        return optim.SparseAdam(params, lr=lr, betas=betas)
    elif opt == 'Adamax':
        return optim.Adamax(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == 'ASGD':
        return optim.ASGD(params, lr=lr, weight_decay=weight_decay)
    elif opt == 'LBFGS':
        return optim.LBFGS(params, lr=lr, max_iter=max_iter)
    elif opt == 'NAdam':
        return optim.NAdam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == 'RAdam':
        return optim.RAdam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == 'RMSprop':
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif opt == 'Rprop':
        return optim.Rprop(params, lr=lr)
    elif opt == 'SGD':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NameError('Wrong Optimizer Type')


def Scheduler(sch, optimizer, base_lr=1e-3, max_lr=1., gamma=1., factor=0.333, lr_lambda=lambda epoch: 1, step_size=100,
              milestones=(10,), epochs=10, steps_per_epoch=100, T_max=1, T_0=1, T_mult=1):
    if type(sch) == str:
        return singleScheduler(sch, optimizer, base_lr, max_lr, gamma, factor, lr_lambda, step_size, milestones, epochs,
                               steps_per_epoch, T_max, T_0, T_mult)
    elif type(sch) == tuple or type(sch) == list:
        schedulers = []
        for s in sch[-1]:
            schedulers.append(
                singleScheduler(s, optimizer, base_lr, max_lr, gamma, factor, lr_lambda, step_size, milestones,
                                epochs, steps_per_epoch, T_max, T_0, T_mult))
        if sch[0] == 'ChainedScheduler':
            return lr_scheduler.ChainedScheduler(schedulers)
        elif sch[0] == 'SequentialLR':
            return lr_scheduler.SequentialLR(optimizer, schedulers, milestones)
        else:
            raise NameError('Wrong lr_Scheduler Type')
    else:
        raise NameError('Wrong lr_Scheduler Type')


def singleScheduler(sch, optimizer, base_lr=1e-3, max_lr=1., gamma=1., factor=0.333, lr_lambda=lambda epoch: 1,
                    step_size=100, milestones=(10,), epochs=10, steps_per_epoch=100, T_max=1, T_0=1, T_mult=1):
    if sch == 'LambdaLR':
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif sch == 'MultiplicativeLR':
        return lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)
    elif sch == 'StepLR':
        return lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
    elif sch == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    elif sch == 'ConstantLR':
        return lr_scheduler.ConstantLR(optimizer, factor=factor, total_iters=milestones[0])
    elif sch == 'LinearLR':
        return lr_scheduler.LinearLR(optimizer, start_factor=gamma, end_factor=factor, total_iters=milestones[0])
    elif sch == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, gamma)
    elif sch == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max)
    elif sch == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor)
    elif sch == 'CyclicLR':
        return lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, gamma=gamma)
    elif sch == 'OneCycleLR':
        return lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    elif sch == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mult)
    else:
        raise NameError('Wrong lr_Scheduler Type')


class LoadClaData(Dataset):
    def __init__(self, dataload, labels):
        self.dataload = dataload
        self.labels = labels

    def __getitem__(self, index):
        return self.dataload[index], self.labels[index].long()

    def __len__(self):
        return self.dataload.shape[0]


class LoadHamData(Dataset):
    def __init__(self, dataload, labels, z):
        self.dataload = dataload
        self.labels = labels
        self.z = z

    def __getitem__(self, index):
        H = self.dataload[index]  # (1, N, N)
        G = torch.linalg.inv(self.z * torch.eye(H.shape[-1], device=H.device) - H) # (1, N, N)
        return G, self.labels[index].long()

    def __len__(self):
        return self.dataload.shape[0]


class LoadHamData_nfz(LoadHamData):
    def __init__(self, dataload, labels, z):
        super(LoadHamData_nfz, self).__init__(dataload, labels, z)

    def __getitem__(self, index):
        H = self.dataload[index]  # (1, N, N)
        zb = self.z[index]
        G = torch.linalg.inv(zb * torch.eye(H.shape[-1], device=H.device) - H) # (1, N, N)
        return G, [self.labels[index].long(), zb]


class LoadHamDatawithH(Dataset):
    def __init__(self, dataload, labels, z):
        self.dataload = dataload
        self.labels = labels
        self.z = z

    def __getitem__(self, index):
        H = self.dataload[index]  # (1, N, N)
        G = torch.linalg.inv(self.z * torch.eye(H.shape[-1], device=H.device) - H) # (1, N, N)
        return [H, G], self.labels[index].long()

    def __len__(self):
        return self.dataload.shape[0]


class LoadHamDatawithH_nfz(LoadHamData):
    def __init__(self, dataload, labels, z):
        super(LoadHamDatawithH_nfz, self).__init__(dataload, labels, z)

    def __getitem__(self, index):
        H = self.dataload[index]  # (1, N, N)
        zb = self.z[index]
        G = torch.linalg.inv(zb * torch.eye(H.shape[-1], device=H.device) - H) # (1, N, N)
        return [H, G], [self.labels[index].long(), zb]


class LoadFKHamData(Dataset):
    def __init__(self, dataload, labels, SEinit):
        self.dataload = dataload
        self.labels = labels
        self.SEinit = SEinit

    def __getitem__(self, index):
        return self.dataload[index], self.SEinit[index], index, self.labels[index]

    def __len__(self):
        return self.dataload.shape[0]


class LoadFKHamDatawithSE(Dataset):
    def __init__(self, dataload, labels, SE):
        self.dataload = dataload
        self.labels = labels
        self.SE = SE

    def __getitem__(self, index):
        return self.dataload[index] + torch.diag_embed(self.SE[index]), self.labels[index, 1:]

    def __len__(self):
        return self.dataload.shape[0]


class CtrlRandomSampler(Sampler[int]):
    data_source: Sized
    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source
        self.my_list = list(range(len(self.data_source)))
        random.shuffle(self.my_list)
        self.num_data = len(self.my_list)

    def __iter__(self) -> Iterator[int]:
        return iter(self.my_list)

    def __len__(self) -> int:
        return self.num_data


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, pin_memory):
        super(MyDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    def _get_iterator(self):
        return MySingleProcessDataLoaderIter(self)


class MySingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self, loader):
        super(MySingleProcessDataLoaderIter, self).__init__(loader)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data, index


def float_or_complex(value):
    if value.startswith('_'):
        value = '-' + value[1:]
    try:
        return float(value)
    except:
        return complex(value)


def None_or_float(value):
    try:
        return float(value)
    except:
        return None


def None_or_complex(value):
    if value.startswith('_'):
        value = '-' + value[1:]
    try:
        return complex(value)
    except:
        return None


def None_or_float_or_complex(value):
    if value.startswith('_'):
        value = '-' + value[1:]
    try:
        return float(value)
    except:
        try:
            return complex(value)
        except:
            return None


def None_or_int(value):
    try:
        return int(value)
    except:
        return None


def str_bool(value):
    if 'True' in value:
        return True
    else:
        return False


def int_or_bool(value):
    try:
        return int(value)
    except:
        return str_bool(value)


def readinfo(filename, anotherfile=None):
    with open(filename, "r") as f:
        cont = f.readlines()
    matchObj = re.match(r'input_size=(.*)\tembedding_size=(.*)\thidden_size=(.*)\toutput_size=(.*)\n', cont[0])
    input_size, embedding_size, output_size = int(matchObj.group(1)), None_or_int(matchObj.group(2)), int(matchObj.group(4))
    hidden_size = tuple(map(None_or_int, matchObj.group(3).split(',')))
    if len(hidden_size) == 1:
        hidden_size = hidden_size[0]
    matchObj = re.match(r'z=(.*)\thermi=(.*)\tdiago=(.*)\trestr=(.*)\tscale=(.*)\treal=(.*)\tdouble=(.*)\n', cont[1])
    if matchObj is None:
        double = False
        matchObj = re.match(r'z=(.*)\thermi=(.*)\tdiago=(.*)\trestr=(.*)\tscale=(.*)\treal=(.*)\n', cont[1])
        if matchObj is None:
            real = False
            matchObj = re.match(r'z=(.*)\thermi=(.*)\tdiago=(.*)\trestr=(.*)\tscale=(.*)\t.*', cont[1])
            if matchObj is None:
                scale = False
                matchObj = re.match(r'z=(.*)\thermi=(.*)\tdiago=(.*)\trestr=(.*)\t.*', cont[1])
            else:
                scale = str_bool(matchObj.group(5))
        else:
            scale, real = str_bool(matchObj.group(5)), str_bool(matchObj.group(6))
    else:
        scale, real, double = str_bool(matchObj.group(5)), str_bool(matchObj.group(6)), str_bool(matchObj.group(7))
    z, hermi= float_or_complex(matchObj.group(1)), int_or_bool(matchObj.group(2))
    diago, restr = matchObj.group(3), matchObj.group(4)
    if diago[0] != '(':
        diago = int_or_bool(diago)
    else:
        diago = tuple(map(int_or_bool, diago[1:-1].split(',')))
    if restr[0] != '(':
        restr = int_or_bool(restr)
    else:
        restr = restr[1:-1]
        left, right = [], []
        for i, s in enumerate(restr):
            if s == '(':
                left.append(i)
            elif s==')':
                right.append(i)
        if len(left) > 0:
            temp = []
            if left[0] > 0:
                temp.extend(list(map(int_or_bool, restr[:left[0] - 2].split(','))))
            for i in range(len(left)):
                temp.append(tuple(map(int_or_bool, restr[left[i] + 1:right[i]].split(','))))
                if i + 1 < len(left) and left[i + 1] - right[i] > 3:
                    temp.extend(list(map(int_or_bool, restr[right[i] + 3:left[i + 1] - 2].split(','))))
            if right[-1] + 1 < len(restr):
                temp.extend(list(map(int_or_bool, restr[right[-1] + 3:].split(','))))
            restr = tuple(temp)
        else:
            restr = tuple(map(int_or_bool, restr.split(',')))
    if anotherfile is not None:
        shutil.copyfile(filename, anotherfile)
        # with open(anotherfile, "w") as f0:
        #     for l in cont:
        #         f0.write(l)
    return input_size, output_size, embedding_size, hidden_size, z, hermi, diago, restr, scale, real, double


def augmentation(crop_size=None, mean_std=None):
    transform = []
    if crop_size is not None:
        transform.append(transforms.CenterCrop(crop_size))
    transform.append(transforms.ToTensor())
    if mean_std is not None:
        transform.append(transforms.Normalize(mean=mean_std[0], std=mean_std[1]))
    return transform


def save_checkpoint(state, is_best, filename, args):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/model_best.pth.tar'.format(args.path))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}={val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        for i, meter in enumerate(self.meters, 1):
            entries.append(str(meter))
            if i % 4 == 0 and i < len(self.meters):
                entries.append('\n')
        # entries += [str(meter) for meter in self.meters]
        data = '\t'.join(entries)
        print(data)
        args.f.write(data + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        if output.dim() == 1:
            assert maxk == 1
            output = torch.vstack((1 - output, output)).t()
            target = target.long()

        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def classes(target, num_classes, origin_classes, double):
    assert num_classes <= origin_classes
    assert origin_classes > 1
    daty = torch.long
    if num_classes == origin_classes:
        return target.long()
    if num_classes == 1:
        if origin_classes == 2:
            return target.double() if double else target.float()
        else:
            num_classes, daty = 2, torch.float64 if double else torch.float32
    gap = round(origin_classes / num_classes)
    if gap * (num_classes - 1) >= origin_classes:
        gap -= 1
    target = torch.vstack((target // gap, (num_classes - 1) * torch.ones(len(target))))
    return torch.min(target, dim=0)[0].type(daty)


def image2H(images, input_size, adj=None, double=False):
    assert input_size == images.shape[-1] ** 2
    if double: images = images.type(torch.complex128)
    images = torch.diag_embed(torch.flatten(images, start_dim=2 if images.dim() == 4 else 1))
    return images if adj is None else images + adj


def imageH2G(images, input_size, z, adj=None, double=False):
    assert input_size == images.shape[-1] ** 2
    images = torch.flatten(images, 2)
    if double: images = images.type(torch.complex128)
    H = torch.diag_embed(images)
    if adj is None:
        if z.dim() == 0:
            return H, torch.diag_embed(1 / (z - images))
        else:
            return H, torch.diag_embed(1 / (z.expand(input_size, 1, -1).transpose(0, 2) - images))
    else:
        H = H + adj
        if z.dim() == 0:
            return H, torch.linalg.inv(torch.diag_embed(z - images) + adj)
        else:
            return H, torch.linalg.inv(torch.diag_embed(z.expand(input_size, 1, -1).transpose(0, 2) - images) + adj)


def image2G(images, input_size, adj=None, z=None, double=False):
    assert input_size == images.shape[-1] ** 2
    images = torch.flatten(images, 2)
    if double: images = images.type(torch.complex128)
    if z is None:
        images = torch.diag_embed(-1j * torch.pi * images)
        return images if adj is None else images + adj
    else:
        if adj is None:
            if z.dim() == 0:
                return torch.diag_embed(1 / (z - images))
            else:
                return torch.diag_embed(1 / (z.expand(input_size, 1, -1).transpose(0, 2) - images))
        else:
            if z.dim() == 0:
                return torch.linalg.inv(torch.diag_embed(z - images) + adj)
            else:
                return torch.linalg.inv(torch.diag_embed(z.expand(input_size, 1, -1).transpose(0, 2) - images) + adj)


def imageG2H(images, z, adj=None, double=False):
    if double: images = images.type(torch.complex128)
    if adj is None:
        return torch.diag_embed(z - 1 / (torch.diagonal(images, dim1=-2, dim2=-1)))
    else:
        return z * torch.eye(images.shape[-1], device=images.device) - torch.linalg.inv(images)


def LossPrepro(output, loss, scale=False):
    if loss == 'NLL':   # scale == True, all output > 0
        output = F.relu(torch.exp(output) - 1.) + 1e-2
        output = torch.log(output / torch.sum(output, dim=1, keepdim=True))
    elif loss == 'BCE':
        if scale:       # scale == True, all output > 0
            output = F.tanh(F.relu(output / 2))
        else:           # scale == False, all output < 0
            output = 1. - F.relu(1. - 2 * F.sigmoid(output))
    return output


def emb_hid_size_wrapper(model):
    module = model.modules()
    next(module)
    size_list = []
    for m in module:
        size_list.append(m.size)
    return size_list[0], tuple(size_list[1:-1])


def compute_grads(module, order=2):
    total_norm, count = 0., 0
    for p in module.parameters():
        count += p.numel()
        param_norm = p.grad.detach().data.norm(order)
        total_norm += param_norm.item() ** order
    total_norm = total_norm ** (1. / order)
    return total_norm, total_norm / count


def myceil(a, precision: int = 0):
    a = np.true_divide(np.ceil(a * 10 ** precision), 10 ** precision)
    return a if precision > 0 else int(a)


def myfloor(a, precision: int = 0):
    a = np.true_divide(np.floor(a * 10 ** precision), 10 ** precision)
    return a if precision > 0 else int(a)


def mymkdir(path):
    if not os.path.exists(path): os.mkdir(path)


def dircheck(data, file, Net):
    path = 'results/{}'.format(data)
    mymkdir(path)
    path = '{}/{}'.format(path, file)
    mymkdir(path)
    path = '{}/{}'.format(path, Net)
    mymkdir(path)
    return path


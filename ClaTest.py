import argparse
import os
import random
import time
import mkl
import warnings

import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.datasets as datasets

from utils import *

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Recursive Green's Function Neural Network")
parser.add_argument('--Net', metavar='NET',
                    help="name of the model which will be trained")
parser.add_argument('--data', metavar='DIR',
                    help='dataset name')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=101, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int, metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--loss', default='CE', type=str, metavar='Loss',
                    help='loss function')
parser.add_argument('--opt', default='SGD', type=str, metavar='OPT',
                    help='optimizer name')
parser.add_argument('--lr', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate')
parser.add_argument('--betas', default=(0.9, 0.999), metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)')
parser.add_argument('--sch', default='StepLR', type=str, metavar='SCH',
                    help='lr_scheduler name')
parser.add_argument('--gamma', default=0.5, type=float,
                    help='gamma rate')
parser.add_argument('--ss', default=20, type=float,
                    help='step size')
parser.add_argument('--gradsnorm', default=False, type=int_or_bool, metavar='N',
                    help='compute gradients norm')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-s', '--save-freq', default=10, type=int, metavar='N',
                    help='save frequency (default: 10)')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

# additional configs:
parser.add_argument('--input_size', type=int, metavar='N',
                    help='input_size')
parser.add_argument('--embedding_size', default=100, type=int)
parser.add_argument('--hidden_size', default=50, type=int)
parser.add_argument('--output_size', default=1, type=int,
                    help='num_classes (default: 2)')
parser.add_argument('--z', default=None, type=None_or_float_or_complex,
                    help='Chemical potential')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
mkl.set_num_threads(1)

best_acc1 = 0.


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.f = open('{}/info.txt'.format(args.path), 'w')
    args.f.write('input_size={}\tembedding_size={}\thidden_size={}\toutput_size={}\tz={}\n'.format(
        args.input_size, args.embedding_size, args.hidden_size, args.output_size, args.z))
    args.f.write('lossfunc={}\topt={}\tlr={}\tbetas={}\twd={}\tsch={}\tgamma={}\tss={}\n'.format(
        args.loss, args.opt, args.lr, args.betas, args.wd, args.sch, args.gamma, args.ss))
    args.f.write('dataset={}\tworkers={}\tepochs={}\tstart_epoch={}\tbatch_size={}\tseed={}\tgradsnorm={}\n'.format(
        args.data, args.workers, args.epochs, args.start_epoch, args.batch_size, args.seed, args.gradsnorm))
    args.f.write('pretrained={}\n'.format(args.pretrained))
    args.f.write('resume={}\n\n'.format(args.resume))
    args.f.flush()
    if args.z is None:
        args.plug = 'cla_'
        args.fixz = True
    elif type(args.z) is float:
        args.z = 1j * args.z
        args.fixz = False
        args.plug = ''
    else:
        args.fixz = True
        args.plug = ''

    args.loss_record, args.accuracy_record = [], []
    if args.gradsnorm:
        args.grads_record, args.agrads_record = [], []
        args.grads1_record, args.agrads1_record = [], []
        args.grads2_record, args.agrads2_record = [], []
        args.grads3_record, args.agrads3_record = [], []
        args.grads4_record, args.agrads4_record = [], []
        args.grads5_record, args.agrads5_record = [], []
        args.grads6_record, args.agrads6_record = [], []
        args.grads_1_record, args.agrads_1_record = [], []

    main_worker(args)

    args.f.write('\nloss={}\n'.format(args.loss_record))
    args.f.write('accuracy={}\n'.format(args.accuracy_record))
    if args.gradsnorm:
        args.f.write('grads={}\n'.format(args.grads_record))
        args.f.write('agrads={}\n'.format(args.agrads_record))
        args.f.write('grads1={}\n'.format(args.grads1_record))
        args.f.write('agrads1={}\n'.format(args.agrads1_record))
        args.f.write('grads2={}\n'.format(args.grads2_record))
        args.f.write('agrads2={}\n'.format(args.agrads2_record))
        args.f.write('grads3={}\n'.format(args.grads3_record))
        args.f.write('agrads3={}\n'.format(args.agrads3_record))
        args.f.write('grads4={}\n'.format(args.grads4_record))
        args.f.write('agrads4={}\n'.format(args.agrads4_record))
        args.f.write('grads5={}\n'.format(args.grads5_record))
        args.f.write('agrads5={}\n'.format(args.agrads5_record))
        args.f.write('grads6={}\n'.format(args.grads6_record))
        args.f.write('agrads6={}\n'.format(args.agrads6_record))
        args.f.write('grads_1={}\n'.format(args.grads_1_record))
        args.f.write('agrads_1={}\n'.format(args.agrads_1_record))

    args.f.close()


def main_worker(args):
    global best_acc1
    print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.Net.startswith('Conv'):
        model = AlexNet(args.input_size, args.output_size, args.embedding_size, args.hidden_size)
    elif args.Net.startswith('Relu'):
        model = Model_relu(args.input_size, args.output_size, args.embedding_size, args.hidden_size)
    else:
        model = Model_sig(args.input_size, args.output_size, args.embedding_size, args.hidden_size)
    print(model)
    args.f.write('{}\n\n'.format(model))
    args.f.flush()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            args.start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    model = model.to(args.device)

    # define loss function (criterion) and optimizer
    if args.loss == 'CE':
        assert args.output_size > 1
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.loss == 'BCEWL':
        assert args.output_size == 1
        criterion = nn.BCEWithLogitsLoss().to(args.device)
    else:
        raise NameError('Wrong Loss Function')


    optimizer = Optimizer(args.opt, model.parameters(), lr=init_lr, weight_decay=args.wd, betas=args.betas)
    if args.lars:
        print("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)
    else:
        scheduler = Scheduler(args.sch, optimizer, gamma=args.gamma, step_size=args.ss)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location="cpu")
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume, map_location=args.device)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if args.data.endswith('MNIST'):
        args.origin_classes = 10
        if args.data == 'MNIST':
            train_dataset = datasets.MNIST('datasets/', train=True, transform=transforms.Compose(augmentation()))
            val_dataset = datasets.MNIST('datasets/', train=False, transform=transforms.Compose(augmentation()))
        else:
            train_dataset = datasets.FashionMNIST('datasets/', train=True, transform=transforms.Compose(augmentation()))
            val_dataset = datasets.FashionMNIST('datasets/', train=False, transform=transforms.Compose(augmentation()))
    elif args.data.startswith('Ins'):
        traindata = torch.load('datasets/{}/train/{}dataset.pt'.format(args.data, args.plug))
        trainlabels = torch.load('datasets/{}/train/labels.pt'.format(args.data))
        valdata = torch.load('datasets/{}/test/{}dataset.pt'.format(args.data, args.plug))
        vallabels = torch.load('datasets/{}/test/labels.pt'.format(args.data))
        if args.fixz and args.data != 'Ins_12':
            traindata = torch.cat((traindata, torch.load('datasets/Ins_12/train/{}dataset.pt'.format(args.plug))), dim=0)
            trainlabels = torch.cat((trainlabels, torch.load('datasets/Ins_12/train/labels.pt')), dim=0)
            valdata = torch.cat((valdata, torch.load('datasets/Ins_12/test/{}dataset.pt'.format(args.plug))), dim=0)
            vallabels = torch.cat((vallabels, torch.load('datasets/Ins_12/test/labels.pt')), dim=0)
        if args.Net.endswith('-'):
            trainlabels = 1 - trainlabels
            vallabels = 1 - vallabels
        if args.z is None:
            args.origin_classes = 2
            train_dataset = LoadClaData(traindata, trainlabels)
            val_dataset = LoadClaData(valdata, vallabels)
        elif args.fixz:
            args.origin_classes = 2
            train_dataset = LoadHamData(traindata, trainlabels, args.z)
            val_dataset = LoadHamData(valdata, vallabels, args.z)
        else:
            args.origin_classes = 3
            trainEs = torch.load('datasets/{}/train/Es.pt'.format(args.data))
            testEs = torch.load('datasets/{}/test/Es.pt'.format(args.data))
            train_dataset = LoadHamData_nfz(traindata, trainlabels, trainEs + args.z)
            val_dataset = LoadHamData_nfz(valdata, vallabels, testEs + args.z)
    else:
        raise NameError('Wrong Dataset')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        # scheduler the learning rate
        if args.lars:
            adjust_learning_rate(optimizer, init_lr, epoch, args)
        else:
            scheduler.step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'Net': args.Net[:args.Net.index('_')],
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, '{}/checkpoint.pth.tar'.format(args.path), args)
        if epoch % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'Net': args.Net[:args.Net.index('_')],
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, False, '{}/checkpoint_{:04d}.pth.tar'.format(args.path, epoch), args)


def augmentation(crop_size=None, mean_std=None):
    transform = []
    if crop_size is not None:
        transform.append(transforms.CenterCrop(crop_size))
    transform.append(transforms.ToTensor())
    if mean_std is not None:
        transform.append(transforms.Normalize(mean=mean_std[0], std=mean_std[1]))
    return transform


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    items = [batch_time, losses, top1]
    if args.gradsnorm:
        grads = AverageMeter('Grad', ':.4e')
        agrads = AverageMeter('AGrad', ':.4e')
        grads1 = AverageMeter('Grad1', ':.4e')
        agrads1 = AverageMeter('AGrad1', ':.4e')
        grads2 = AverageMeter('Grad2', ':.4e')
        agrads2 = AverageMeter('AGrad2', ':.4e')
        grads3 = AverageMeter('Grad3', ':.4e')
        agrads3 = AverageMeter('AGrad3', ':.4e')
        grads4 = AverageMeter('Grad4', ':.4e')
        agrads4 = AverageMeter('AGrad4', ':.4e')
        grads5 = AverageMeter('Grad5', ':.4e')
        agrads5 = AverageMeter('AGrad5', ':.4e')
        grads6 = AverageMeter('Grad6', ':.4e')
        agrads6 = AverageMeter('AGrad6', ':.4e')
        grads_1 = AverageMeter('Grad-1', ':.4e')
        agrads_1 = AverageMeter('AGrad-1', ':.4e')
        items += [grads, agrads, agrads1, agrads2, agrads_1]
    progress = ProgressMeter(len(train_loader), items, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.data.endswith('MNIST'):
            if not args.Net.startswith('Conv'):
                images = torch.flatten(images, 1)
        else:
            if args.z is not None:
                images = torch.view_as_real(images)
            if args.Net.startswith('Conv'):
                images = images.transpose(1, -1).squeeze(-1)
            else:
                images = torch.flatten(images, 1)

        if not args.fixz:
            target = target[0]
        target = classes(target, args.output_size, args.origin_classes)

        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        # compute output
        output = model(images)
        # print(output[0])
        output = LossPrepro(output, args.loss)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.gradsnorm:
            total_norm, avg_norm = compute_grads(model, order=args.gradsnorm)
            grads.update(total_norm, images.size(0))
            agrads.update(avg_norm, images.size(0))
            total_norm, avg_norm = compute_grads(model.pre, order=args.gradsnorm)
            grads1.update(total_norm, images.size(0))
            agrads1.update(avg_norm, images.size(0))
            total_norm, avg_norm = compute_grads(model.emb, order=args.gradsnorm)
            grads2.update(total_norm, images.size(0))
            agrads2.update(avg_norm, images.size(0))
            total_norm, avg_norm = compute_grads(model.fc1, order=args.gradsnorm)
            grads3.update(total_norm, images.size(0))
            agrads3.update(avg_norm, images.size(0))
            total_norm, avg_norm = compute_grads(model.fc2, order=args.gradsnorm)
            grads4.update(total_norm, images.size(0))
            agrads4.update(avg_norm, images.size(0))
            total_norm, avg_norm = compute_grads(model.fc3, order=args.gradsnorm)
            grads5.update(total_norm, images.size(0))
            agrads5.update(avg_norm, images.size(0))
            total_norm, avg_norm = compute_grads(model.fc4, order=args.gradsnorm)
            grads6.update(total_norm, images.size(0))
            agrads6.update(avg_norm, images.size(0))
            total_norm, avg_norm = compute_grads(model.out, order=args.gradsnorm)
            grads_1.update(total_norm, images.size(0))
            agrads_1.update(avg_norm, images.size(0))
        if 'no' not in args.Net: nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)

    args.loss_record.append(losses.avg)
    if args.gradsnorm:
        args.grads_record.append(float('{:.4e}'.format(grads.avg)))
        args.agrads_record.append(float('{:.4e}'.format(agrads.avg)))
        args.grads1_record.append(float('{:.4e}'.format(grads1.avg)))
        args.agrads1_record.append(float('{:.4e}'.format(agrads1.avg)))
        args.grads2_record.append(float('{:.4e}'.format(grads2.avg)))
        args.agrads2_record.append(float('{:.4e}'.format(agrads2.avg)))
        args.grads3_record.append(float('{:.4e}'.format(grads3.avg)))
        args.agrads3_record.append(float('{:.4e}'.format(agrads3.avg)))
        args.grads4_record.append(float('{:.4e}'.format(grads4.avg)))
        args.agrads4_record.append(float('{:.4e}'.format(agrads4.avg)))
        args.grads5_record.append(float('{:.4e}'.format(grads5.avg)))
        args.agrads5_record.append(float('{:.4e}'.format(agrads5.avg)))
        args.grads6_record.append(float('{:.4e}'.format(grads6.avg)))
        args.agrads6_record.append(float('{:.4e}'.format(agrads6.avg)))
        args.grads_1_record.append(float('{:.4e}'.format(grads_1.avg)))
        args.agrads_1_record.append(float('{:.4e}'.format(agrads_1.avg)))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.data.endswith('MNIST'):
                if not args.Net.startswith('Conv'):
                    images = torch.flatten(images, 1)
            else:
                if args.z is not None:
                    images = torch.view_as_real(images)
                if args.Net.startswith('Conv'):
                    images = images.transpose(1, -1).squeeze(-1)
                else:
                    images = torch.flatten(images, 1)

            if not args.fixz:
                target = target[0]
            target = classes(target, args.output_size, args.origin_classes)

            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            # compute output
            output = model(images).squeeze()
            output = LossPrepro(output, args.loss)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, args)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        args.f.write(' * Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
        args.f.flush()

        args.accuracy_record.append(top1.avg)

    return top1.avg


class Model_relu(nn.Module):
    def __init__(self, input_size, output_size=2, embedding_size=100, hidden_size=64):
        super(Model_relu, self).__init__()
        self.pre = nn.Linear(input_size, embedding_size)
        self.emb = nn.Linear(embedding_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.pre.apply(self.weights_init)
        self.emb.apply(self.weights_init)
        self.fc1.apply(self.weights_init)
        self.fc2.apply(self.weights_init)
        self.fc3.apply(self.weights_init)
        self.fc4.apply(self.weights_init)
        self.out.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            nn.init.kaiming_uniform_(m.weight)
            # nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.pre(x))
        x = F.relu(self.emb(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x).squeeze(1)


class Model_sig(nn.Module):
    def __init__(self, input_size, output_size=2, embedding_size=100, hidden_size=64):
        super(Model_sig, self).__init__()
        self.pre = nn.Linear(input_size, embedding_size)
        self.emb = nn.Linear(embedding_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.pre.apply(self.weights_init)
        self.emb.apply(self.weights_init)
        self.fc1.apply(self.weights_init)
        self.fc2.apply(self.weights_init)
        self.fc3.apply(self.weights_init)
        self.fc4.apply(self.weights_init)
        self.out.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            nn.init.kaiming_uniform_(m.weight)
            # nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.sigmoid(self.pre(x))
        x = F.sigmoid(self.emb(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return self.out(x).squeeze(1)


class AlexNet(nn.Module):
    def __init__(self, input_size=2, output_size=10, embedding_size=4, hidden_size=2, dropout: float = 0.5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=11, stride=embedding_size, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=hidden_size),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(1)


if __name__ == '__main__':
    path = 'models/{}'.format(args.data)
    mymkdir(path)
    path = '{}/{}'.format(path, args.Net)
    mymkdir(path)
    args.path = path

    args.betas = tuple(map(float, args.betas.split(',')))

    main(args)

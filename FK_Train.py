import argparse
import os
import random
import time
import mkl
import warnings

import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from utils import *
from FK_rgfnn import Network
from ent import computeI, getIndex
from FK_DMFT import DMFT

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Recursive Green's Function Neural Network")
parser.add_argument('-t', '--threads', default=8, type=int, metavar='N',
                    help='number of threads (default: 8)')
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
parser.add_argument('--ss', default=20, type=int,
                    help='step size')
parser.add_argument('--drop', default=0., type=float,
                    help='dropout')
parser.add_argument('--disor', default=0., type=float,
                    help='disorder')
parser.add_argument('--gradsnorm', default=False, type=int_or_bool, metavar='N',
                    help='compute gradients norm')
parser.add_argument('--entanglement', default=False,
                    help='compute mutual information')
parser.add_argument('--delta', default=1e-2, type=float,
                    help='differential of mutual information')
parser.add_argument('--tc', default=None, type=None_or_int, metavar='N',
                    help='target category')
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
parser.add_argument('--input_size', type=int, metavar='N')
parser.add_argument('--embedding_size', type=None_or_int)
parser.add_argument('--hidden_size')
parser.add_argument('--output_size', type=int,
                    help='num_classes')
parser.add_argument('--hermi', default=True, type=int_or_bool,
                    help='Hermitian h')
parser.add_argument('--diago', default=False,
                    help='Diagonal h')
parser.add_argument('--restr', default=False,
                    help='Restricted t (0: False, 1: 1D, 2: 2D')
parser.add_argument('--real', action='store_true',
                    help='All parameters are real')
parser.add_argument('--init_bound', default=1., type=float,
                    help='Initial bound')
parser.add_argument('--scale', action='store_true',
                    help='True: all output > 0')
parser.add_argument('--double', action='store_true',
                    help='dtype = float / double')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')

# DMFT configs:
parser.add_argument('--Tem', default=1e-3, type=float,
                    help='Temperature')
parser.add_argument('--count', default=100, type=int,
                    help='n count')
parser.add_argument('--iota', default=1e-3, type=float,
                    help='iota')
parser.add_argument('--momentum', default=0., type=float,
                    help='momentum')
parser.add_argument('--maxEpoch', default=100, type=int,
                    help='max epoch')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
mkl.set_num_threads(args.threads)

best_acc1 = 0.


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sanity check and reset
    if args.loss == 'CE':
        assert args.output_size > 1
    elif args.loss == 'NLL':
        assert args.output_size > 1
        args.scale = True
    else:
        args.output_size = 1
        if args.loss == 'BCEWL':
            assert args.Net.startswith('C')
            if not args.Net.startswith('Classical'):
                args.scale = False

    args.f = open('{}/info.txt'.format(args.path), 'w')
    args.f.write('input_size={}\tembedding_size={}\thidden_size={}\toutput_size={}\n'.format(
        args.input_size, args.embedding_size, args.hidden_size, args.output_size))
    args.f.write('z={}\thermi={}\tdiago={}\trestr={}\tscale={}\treal={}\tdouble={}\n'.format(
        None, args.hermi, args.diago, args.restr, args.scale, args.real, args.double))
    args.f.write('T={}\tcount={}\tiota={}\tmomentum={}\tmaxEpoch={}\n'.format(
        args.Tem, args.count, args.iota, args.momentum, args.maxEpoch))
    args.f.write('dataset={}\tentanglement={}\tdelta={}\ttc={}\tgradsnorm={}\n'.format(
        args.data, args.entanglement, args.delta, args.tc, args.gradsnorm))
    args.f.write('lossfunc={}\topt={}\tlr={}\tbetas={}\twd={}\tlars={}\tdevice={}\n'.format(
        args.loss, args.opt, args.lr, args.betas, args.wd, args.lars, args.device))
    args.f.write('sch={}\tgamma={}\tstep_size={}\tinit_bound={}\tdrop={}\tdisor={}\n'.format(
        args.sch, args.gamma, args.ss, args.init_bound, args.drop, args.disor))
    args.f.write('workers={}\tepochs={}\tstart_epoch={}\tbatch_size={}\tseed={}\n'.format(
        args.workers, args.epochs, args.start_epoch, args.batch_size, args.seed))
    args.f.write('pretrained={}\n'.format(args.pretrained))
    args.f.write('resume={}\n\n'.format(args.resume))
    args.f.flush()

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
    if args.entanglement:
        args.f.write('Iacd={}\n'.format(args.Iacd_record))
        args.f.write('Iac={}\n'.format(args.Iac_record))
        args.f.write('Iad={}\n'.format(args.Iad_record))
        args.f.write('Ia_cd={}\n'.format(args.Ia_cd_record))
        if not None in args.index:
            args.f.write('Ibcd={}\n'.format(args.Ibcd_record))
            args.f.write('Ibc={}\n'.format(args.Ibc_record))
            args.f.write('Ibd={}\n'.format(args.Ibd_record))
            args.f.write('Ib_cd={}\n'.format(args.Ib_cd_record))

    args.f.close()


def main_worker(args):
    global best_acc1
    print("Use GPU: {} for training".format(args.gpu))
    scf = DMFT(args.Tem, args.count, args.iota, args.momentum, args.maxEpoch, args.device, args.double)

    # create model
    Net = args.Net[:args.Net.index('_')]
    model = Network(Net, args.input_size, args.output_size, args.embedding_size, args.hidden_size, scf.iomega, args.hermi,
                    args.diago, args.restr, args.real, args.init_bound, args.scale, args.drop, args.disor, args.double)
    print(model)
    args.f.write('{}\n\n'.format(model))
    args.f.flush()

    if args.entanglement:
        args.embedding_size, args.hidden_size = emb_hid_size_wrapper(model)
        args.index = getIndex(args.input_size, args.embedding_size, args.hidden_size, args.output_size, args.tc)
        args.Iacd_record, args.Iac_record, args.Iad_record, args.Ia_cd_record = [], [], [], []
        if not None in args.index:
            args.Ibcd_record, args.Ibc_record, args.Ibd_record, args.Ib_cd_record = [], [], [], []

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            args.start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'], strict=False)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    model = model.to(args.device)

    # define loss function (criterion) and optimizer
    if args.loss == 'NLL':
        criterion = nn.NLLLoss().to(args.device)
    elif args.loss == 'CE':
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.loss == 'BCE':
        criterion = nn.BCELoss().to(args.device)
    elif args.loss == 'BCEWL':
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
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindata = torch.load('datasets/{}/train/dataset.pt'.format(args.data))
    trainlabels = torch.load('datasets/{}/train/labels.pt'.format(args.data))
    valdata = torch.load('datasets/{}/test/dataset.pt'.format(args.data))
    vallabels = torch.load('datasets/{}/test/labels.pt'.format(args.data))
    if args.double:
        traindata, valdata = traindata.type(torch.complex128), valdata.type(torch.complex128)
    if args.Net.endswith('-'):
        trainlabels = 1 - trainlabels
        vallabels = 1 - vallabels
    train_dataset = LoadFKHamData(traindata, trainlabels)
    val_dataset = LoadFKHamData(valdata, vallabels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=args.workers, pin_memory=True)
    print('trainset:{}\ttrainloader:{}\tvalset:{}\tvalloader:{}'.format(
        len(train_dataset), len(train_loader), len(val_dataset), len(val_loader)))
    args.f.write('trainset:{}\ttrainloader:{}\tvalset:{}\tvalloader:{}\n\n'.format(
        len(train_dataset), len(train_loader), len(val_dataset), len(val_loader)))
    args.f.flush()

    for epoch in range(args.start_epoch, args.epochs):
        # scheduler the learning rate
        if args.lars:
            adjust_learning_rate(optimizer, init_lr, epoch, args)
        else:
            scheduler.step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scf, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, scf, args)

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


def train(train_loader, model, criterion, optimizer, scf, epoch, args):
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
    if args.entanglement:
        Iac = AverageMeter('Iac', ':.4e')
        if args.full:
            Iacd = AverageMeter('Iacd', ':.4e')
            Iad = AverageMeter('Iad', ':.4e')
            Ia_cd = AverageMeter('Ia_cd', ':.4e')
            items += [Iacd, Iac, Iad, Ia_cd]
        else:
            items += [Iac]
        if not None in args.index:
            Ibc = AverageMeter('Ibc', ':.4e')
            if args.full:
                Ibcd = AverageMeter('Ibcd', ':.4e')
                Ibd = AverageMeter('Ibd', ':.4e')
                Ib_cd = AverageMeter('Ib_cd', ':.4e')
                items += [Ibcd, Ibc, Ibd, Ib_cd]
            else:
                items += [Ibc]
    progress = ProgressMeter(len(train_loader), items, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (H0, E_mu, U, target) in enumerate(train_loader):
        bz = H0.size(0)
        H0 = H0.to(args.device, non_blocking=True)
        U = U.to(args.device, non_blocking=True)
        E_mu = E_mu.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        H = H0 + torch.diag_embed(scf(H0, E_mu, U, model))  # (bz, count, size, size)

        # compute output
        output = model(H)
        # print(output[0])
        output = LossPrepro(output, args.loss, args.scale)
        loss = criterion(output, target)

        if args.entanglement and i % args.print_freq == 0:
            if args.tc is None:
                He = H.cpu()
            else:
                index = torch.nonzero(target == args.tc, as_tuple=True)
                if len(index[0]) > 0:
                    He = H[index].cpu()
            try:
                Ia, Ib = computeI(model, args.input_size, He, args.index, args.entanglement, args.delta,
                                  double=args.double, full=args.full)
                Ia = torch.mean(Ia, dim=1).numpy()
                if args.full:
                    Iacd.update(Ia[0], bz)
                    Iac.update(Ia[1], bz)
                    Iad.update(Ia[2], bz)
                    Ia_cd.update(Ia[3], bz)
                else:
                    Iac.update(Ia[0], bz)
                if Ib is not None:
                    Ib = torch.mean(Ib, dim=1).numpy()
                    if args.full:
                        Ibcd.update(Ib[0], bz)
                        Ibc.update(Ib[1], bz)
                        Ibd.update(Ib[2], bz)
                        Ib_cd.update(Ib[3], bz)
                    else:
                        Ibc.update(Ib[0], bz)
            except NameError:
                pass

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), bz)
        top1.update(acc1[0], bz)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.gradsnorm:
            total_norm, avg_norm = compute_grads(model, order=args.gradsnorm)
            grads.update(total_norm, bz)
            agrads.update(avg_norm, bz)
            total_norm, avg_norm = compute_grads(model.pre, order=args.gradsnorm)
            grads1.update(total_norm, bz)
            agrads1.update(avg_norm, bz)
            total_norm, avg_norm = compute_grads(model.emb, order=args.gradsnorm)
            grads2.update(total_norm, bz)
            agrads2.update(avg_norm, bz)
            if (not args.Net.startswith('Naive')) or (not args.Net.startswith('CNaive')):
                total_norm, avg_norm = compute_grads(model.fc1, order=args.gradsnorm)
                grads3.update(total_norm, bz)
                agrads3.update(avg_norm, bz)
                total_norm, avg_norm = compute_grads(model.fc2, order=args.gradsnorm)
                grads4.update(total_norm, bz)
                agrads4.update(avg_norm, bz)
            if args.Net.startswith('Median') or args.Net.startswith('CMedian'):
                total_norm, avg_norm = compute_grads(model.fc3, order=args.gradsnorm)
                grads5.update(total_norm, bz)
                agrads5.update(avg_norm, bz)
                total_norm, avg_norm = compute_grads(model.fc4, order=args.gradsnorm)
                grads6.update(total_norm, bz)
                agrads6.update(avg_norm, bz)
            total_norm, avg_norm = compute_grads(model.out, order=args.gradsnorm)
            grads_1.update(total_norm, bz)
            agrads_1.update(avg_norm, bz)
        if 'no' not in args.Net: nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)

    args.loss_record.append(float('{:.4e}'.format(losses.avg)))
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
    if args.entanglement:
        args.Iac_record.append(float('{:.4e}'.format(Iac.avg)))
        if args.full:
            args.Iacd_record.append(float('{:.4e}'.format(Iacd.avg)))
            args.Iad_record.append(float('{:.4e}'.format(Iad.avg)))
            args.Ia_cd_record.append(float('{:.4e}'.format(Ia_cd.avg)))
        else:
            args.Iacd_record.append(0.)
            args.Iad_record.append(0.)
            args.Ia_cd_record.append(0.)
        if not None in args.index:
            args.Ibc_record.append(float('{:.4e}'.format(Ibc.avg)))
            if args.full:
                args.Ibcd_record.append(float('{:.4e}'.format(Ibcd.avg)))
                args.Ibd_record.append(float('{:.4e}'.format(Ibd.avg)))
                args.Ib_cd_record.append(float('{:.4e}'.format(Ib_cd.avg)))
            else:
                args.Ibcd_record.append(0.)
                args.Ibd_record.append(0.)
                args.Ib_cd_record.append(0.)


def validate(val_loader, model, criterion, scf, args):
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
        for i, (H0, E_mu, U, target) in enumerate(val_loader):
            bz = H0.size(0)
            H0 = H0.to(args.device, non_blocking=True)
            U = U.to(args.device, non_blocking=True)
            E_mu = E_mu.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            H = H0 + torch.diag_embed(scf(H0, E_mu, U, model))  # (bz, count, size, size)

            # compute output
            output = model(H)
            output = LossPrepro(output, args.loss, args.scale)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), bz)
            top1.update(acc1[0], bz)

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


if __name__ == '__main__':
    args.hidden_size = tuple(map(None_or_int, args.hidden_size.split(',')))
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]
    args.betas = tuple(map(float, args.betas.split(',')))
    args.diago = tuple(map(int_or_bool, args.diago.split(',')))
    if len(args.diago) == 1:
        args.diago = args.diago[0]
    args.restr = args.restr.split(',')
    for i in range(len(args.restr)):
        args.restr[i] = tuple(map(int_or_bool, args.restr[i].split('_')))
        if len(args.restr[i]) == 1: args.restr[i] = args.restr[i][0]
    if len(args.restr) == 1:
        args.restr = args.restr[0]
    else:
        args.restr = tuple(args.restr)
    try:
        args.entanglement = int(args.entanglement)
        args.full = True
    except:
        try:
            args.entanglement = int(float(args.entanglement))
            args.full = False
        except:
            args.entanglement = False

    path = 'models/{}'.format(args.data)
    mymkdir(path)
    path = '{}/{}'.format(path, args.Net)
    mymkdir(path)
    args.path = path

    main(args)

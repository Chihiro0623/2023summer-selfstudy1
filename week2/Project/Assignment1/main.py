
import argparse
import time

parser = argparse.ArgumentParser(description='Assignment 1')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='Workers',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='E',
                    help='number of total epochs to run (default: 10)')
parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='B',
                    help='mini-batch size (default: 16)')
parser.add_argument('-tb', '--test_batch_size', default=2, type=int, metavar='TB',
                    help='test batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float, metavar='LR',
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--gpu_id', '--gid', type=str, metavar='gpuid',
                    help='gpu id')
parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')

args = parser.parse_args()

log_interval = 5000 // args.batch_size

print(args.gpu_id)

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import CustomDataset, CustomModel
import numpy as np

cifar100trainpath = '/home/oso0310/private/cifar-100-python/train'
cifar100testpath = '/home/oso0310/private/cifar-100-python/test'
modelsavepath = '/home/oso0310/private/project1/savedmodels'
bestmodelpath = '/home/oso0310/private/project1/Assignment1'
timestamp = 0

def initwandb():
    wandb.init(
        # set the wandb project where this run will be logged
        project="project1",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.01,
            "architecture": "ResNeXt",
            "dataset": "CIFAR-100",
            "epochs": 300,
        }
    )


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k)
        # res.append(wrong_k.mul_(100.0 / batch_size))

    return res


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, device_id, world_size):
    global timestamp
    model.train()
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()

        optimizer.zero_grad()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            output = model(input)
            loss = criterion(output, target)

        loss = reduce_tensor(loss, world_size)
        loss.backward()

        optimizer.step()

        if device_id == 0:
            if batch_idx % log_interval == 0:
                wandb.log({"train loss": loss})
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime : {:.6f}'.format(
                    epoch, batch_idx * len(input), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data, time.time()-timestamp))


def test(test_loader, model, criterion, device_id, world_size):
    global timestamp
    model.eval()
    test_loss = 0
    correct = 0
    err1sum = 0
    err5sum = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()

        output = model(data)
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        err1sum += err1
        err5sum += err5

        test_loss += criterion(output, target).data

    test_loss /= len(test_loader.dataset)
    test_loss = reduce_tensor(test_loss, world_size)
    err1sum = reduce_tensor(err1sum, world_size) * world_size

    if device_id == 0:
        wandb.log({"test loss": test_loss, "top1acc": 10000 - err1sum,  "top5acc": 10000-err5sum, "top1err": err1sum,  "top5err": err5sum, "time": time.time()-timestamp})
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def main():
    global timestamp

    if os.environ.get('WORLD_SIZE', None):
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        device = torch.cuda.current_device()
        world_size = dist.get_world_size()
        timestamp = time.time()
        device_id = rank % torch.cuda.device_count()
        distributed = True
    else:
        rank = 0
        device = 'cuda:0' # torch.cuda.current_device()
        world_size = 1 # dist.get_world_size()
        device_id = 0
        distributed=False

    if device_id == 0:
        initwandb()

    model = CustomModel.ResNeXt152().cuda()

    if distributed:
        ddp_model = DDP(model, device_ids=[device_id], output_device=device)
        ddp_model.load_state_dict(torch.load(bestmodelpath, map_location='cpu').state_dict())
    else:
        result = model.load_state_dict(torch.load(bestmodelpath, map_location='cpu').state_dict())
        print(f"load state_dict...{result}")
        ddp_model = model

    # from deepspeed.profiling.flops_profiler import get_model_profile
    # from deepspeed.accelerator import get_accelerator
    #
    # flops, macs, params = get_model_profile(model=model,  # model
    #                                         input_shape=(1, 3, 224, 224),
    #                                         # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
    #                                         args=None,  # list of positional arguments to the model.
    #                                         kwargs=None,  # dictionary of keyword arguments to the model.
    #                                         print_profile=True,
    #                                         # prints the model graph with the measured profile attached to each module
    #                                         detailed=True,  # print the detailed profile
    #                                         module_depth=-1,
    #                                         # depth into the nested modules, with -1 being the inner most modules
    #                                         top_modules=1,  # the number of top modules to print aggregated profile
    #                                         warm_up=10,
    #                                         # the number of warm-ups before measuring the time of each module
    #                                         as_string=True,
    #                                         # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
    #                                         output_file=None,
    #                                         # path to the output file. If None, the profiler prints to stdout.
    #                                         ignore_modules=None)  # the list of modules to ignore in the profiling
    # print(flops, macs, params)
    train_data = CustomDataset.CustomDataset(cifar100trainpath, train=True, transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
    ]))
    test_data = CustomDataset.CustomDataset(cifar100testpath, train=False, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=4, sampler=test_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                   sampler=None)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=4, sampler=None)

    # optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(ddp_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(1, args.epochs + 1):
        # train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch)

        # train(train_loader, ddp_model, criterion, optimizer, epoch, device_id, world_size)
        test(test_loader, ddp_model, criterion, device_id, world_size)
        # torch.save(model.state_dict(), f"{modelsavepath}/{epoch}")

if __name__ == '__main__':
    main()
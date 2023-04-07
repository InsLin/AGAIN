from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from utils import Bar, Logger, AverageMeter, accuracy
from utils_awp import MixAWP


parser = argparse.ArgumentParser(description='PyTorch CIFAR AGAIN_AWP Adversarial Training')
parser.add_argument('--arch', type=str, default='WideResNet34')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data-path', type=str, default='./data/cifar10',
                    help='where is the dataset CIFAR-10')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'],
                    help='The threat model')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./ckpt',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim', default='', type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--save-freq', '-s', default=20, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise')
parser.add_argument('--awp-warmup', default=10, type=int,
                    help='We could apply AWP after some epochs for accelerating.')

args = parser.parse_args()
epsilon = args.epsilon / 255
step_size = args.step_size / 255
if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty
if args.data == 'CIFAR100':
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = getattr(datasets, args.data)(
    root=args.data_path, train=True, download=True, transform=transform_train)
testset = getattr(datasets, args.data)(
    root=args.data_path, train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def perturb_input(model,
                  x_natural,
                  y,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf'):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            
            with torch.enable_grad():
                outputs, class_wise_output = model(x_adv)
                loss = F.cross_entropy(outputs, y)
                channel_reg_loss = 0.
                for extra_output in class_wise_output:
                    channel_reg_loss += F.cross_entropy(extra_output, y)
                if len(class_wise_output) > 0:
                    channel_reg_loss /= len(class_wise_output)
                loss += 2 * channel_reg_loss
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def train(model, train_loader, optimizer, epoch, awp_adversary):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))

    for batch_idx, (data, target) in enumerate(train_loader):
        x_natural, target = data.to(device), target.to(device)

        # craft adversarial examples
        x_adv = perturb_input(model=model,
                              x_natural=x_natural,
                              y=target,
                              step_size=step_size,
                              epsilon=epsilon,
                              perturb_steps=args.num_steps,
                              distance=args.norm)

        model.train()
        # calculate adversarial weight perturbation
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv, targets=target)
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        
        output, class_wise_output= model(x_adv, target)
        loss = F.cross_entropy(output, target)
        channel_reg_loss = 0.
        for extra_output in class_wise_output:
            channel_reg_loss += F.cross_entropy(extra_output, target)
        channel_reg_loss = channel_reg_loss / len(class_wise_output)
        loss = loss + 2 * channel_reg_loss

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))

        # update the parameters at last
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def test(model, test_loader, criterion):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            output, class_wise_output= model(inputs)
            loss = F.cross_entropy(output, targets)
            channel_reg_loss = 0.
            for extra_output in class_wise_output:
                channel_reg_loss += F.cross_entropy(extra_output, targets)
            channel_reg_loss = channel_reg_loss / len(class_wise_output)
            loss = loss + 2 * channel_reg_loss

            prec1, prec5 = accuracy(output.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}'.format(
                batch=batch_idx + 1,
                size=len(test_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg)
            bar.next()
    bar.finish()
    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return 

def main():
    # init model, ResNet18() can be also used here for training
    model = models.WideResNet34().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # We use a proxy model to calculate AWP, which does not affect the statistics of BN.
    proxy = models.WideResNet34().to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = MixAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()

    logger = Logger(os.path.join(model_dir, 'log.txt'), title=args.arch)
    logger.set_names(['Learning Rate',
                      'Adv Train Loss', 'Nat Train Loss', 'Nat Val Loss',
                      'Adv Train Acc.', 'Nat Train Acc.', 'Nat Val Acc.'])

    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
    if args.resume_optim:
        optimizer.load_state_dict(torch.load(args.resume_optim, map_location=device))

    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer, epoch)
        
        # adversarial training
        adv_loss, adv_acc = train(model, train_loader, optimizer, epoch, awp_adversary)

        # evaluation on natural examples
        print('================================================================')
        train_loss, train_acc = test(model, train_loader, criterion)
        val_loss, val_acc = test(model, test_loader, criterion)
        print('================================================================')
        logger.append([lr, adv_loss, train_loss, val_loss, adv_acc, train_acc, val_acc])

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'ours-model-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'ours-opt-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()

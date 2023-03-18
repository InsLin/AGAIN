import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import models
import attack_generator as attack

parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--data', type=str, default="cifar10", help="choose from cifar10,cifar100")
parser.add_argument('--attack_method', type=str,default="dat", help = "choose form: dat and trades")
parser.add_argument('--model_path', default='./ckpt', help='model for white-box attack evaluation')
parser.add_argument('--method',type=str,default='dat',help='select attack setting following DAT or TRADES')

args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.data == "cifar10":
    testset = torchvision.datasets.CIFAR10(root='../../dynamic/data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.data == "cifar100":
    testset = torchvision.datasets.CIFAR10(root='../data/cifar100', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

if args.data == 'CIFAR100':
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

print('==> Load Model')
model = models.WideResNet34().cuda()
model.load_state_dict(torch.load(args.model_path))

print('==> Evaluating Performance under White-box Adversarial Attack')

loss, test_nat_acc = attack.eval_clean(model, test_loader)
print('Natural Test Accuracy: {:.2f}%'.format(100. * test_nat_acc))
loss, fgsm_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,loss_fn="cent", rand_init=True)
print('FGSM Test Accuracy: {:.2f}%'.format(100. * fgsm_acc))
loss, pgd10_acc = attack.eval_robust(model, test_loader, perturb_steps=10, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", rand_init=True)
print('PGD10 Test Accuracy: {:.2f}%'.format(100. * pgd10_acc))
loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", rand_init=True)
print('PGD20 Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))
loss, pgd50_acc = attack.eval_robust(model, test_loader, perturb_steps=50, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", rand_init=True)
print('PGD50 Test Accuracy: {:.2f}%'.format(100. * pgd50_acc))
loss, pgd100_acc = attack.eval_robust(model, test_loader, perturb_steps=100, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", rand_init=True)
print('PGD100 Test Accuracy: {:.2f}%'.format(100. * pgd100_acc))
loss, cw_acc = attack.eval_robust(model, test_loader, perturb_steps=30, epsilon=0.031, step_size=0.031 / 4,loss_fn="cw", rand_init=True)
print('CW Test Accuracy: {:.2f}%'.format(100. * cw_acc))


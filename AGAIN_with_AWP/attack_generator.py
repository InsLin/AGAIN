import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

def cwloss(output, target,confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def pgd(model, data, target, epsilon, step_size, num_steps, loss_fn, rand_init):
    model.eval()
    x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
#     random_noise = torch.FloatTensor(*data.shape).uniform_(-epsilon, epsilon).cuda()
#     x_adv = Variable(data.data + random_noise, requires_grad=True)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output, class_wise_output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
                channel_reg_loss = 0.
                for extra_output in class_wise_output:
                    channel_reg_loss += nn.CrossEntropyLoss(reduction="mean")(extra_output, target)
                if len(class_wise_output) > 0:
                    channel_reg_loss /= len(class_wise_output)
                loss_adv += 2 * channel_reg_loss
                
            if loss_fn == "cw":
                loss_adv = cwloss(output,target) 
                channel_reg_loss = 0.
                for extra_output in class_wise_output:
                    channel_reg_loss += cwloss(extra_output, target)
                if len(class_wise_output) > 0:
                    channel_reg_loss /= len(class_wise_output)
                loss_adv += 2 * channel_reg_loss
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_loader)):
            data, target = batch[0].cuda(), batch[1].cuda()
            output,_ = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Natrual Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, rand_init):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for _, batch in tqdm(enumerate(test_loader)):
            data, target = batch[0].cuda(), batch[1].cuda()
            x_adv = pgd(model,data,target,epsilon,step_size,perturb_steps,loss_fn,rand_init=rand_init)
            output, _ = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Attack Setting ==> Loss_fn:{}, Perturb steps:{}, Epsilon:{}, Step dize:{} \n Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(loss_fn,perturb_steps,epsilon,step_size,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


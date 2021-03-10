# -*- coding: utf-8 -*-
import torch
import math
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required


def collect_params(model_list, exclude_bias_and_bn=True):
    """
    exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
        in the PyTorch implementation of ResNet, `downsample.1` are bn layers
    """
    param_list = []
    for model in model_list:
        for name, param in model.named_parameters():
            if exclude_bias_and_bn and ('bn' in name or 'downsample.1' in name or 'bias' in name):
                param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
            else:
                param_dict = {'params': param}
            param_list.append(param_dict)
    return param_list


class LARSSGD(Optimizer):
    """
    Layer-wise adaptive rate scaling

    https://github.com/yaox12/BYOL-PyTorch

    - Based on:

    https://github.com/noahgolmant/pytorch-lars

    params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)

        lr (int): Length / Number of layers we want to apply weight decay, else do not compute

        momentum (float, optional): momentum factor (default: 0.9)

        nesterov (bool, optional): flag to use nesterov momentum (default: False)

        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
            ("\beta")

        eta (float, optional): LARS coefficient (default: 0.001)

    - Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.

    - Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888

    """

    def __init__(self,
                 params,
                 lr,
                 momentum=0.9,
                 dampening=0.0,
                 weight_decay=0.0,
                 eta=0.001,
                 nesterov=False):

        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov, eta=eta)

        super(LARSSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARSSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            eta = group['eta']
            nesterov = group['nesterov']
            lr = group['lr']
            lars_exclude = group.get('lars_exclude', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                p_grad = p.grad

                if lars_exclude:
                    learning_rate = 1.0
                else:
                    weight_norm = torch.norm(p).item()
                    grad_norm = torch.norm(p_grad).item()

                    learning_rate = eta * weight_norm / ((grad_norm + weight_decay * weight_norm) + 1E-12)

                scaled_learning_rate = learning_rate * lr

                p_grad = p_grad.add(p, alpha=weight_decay).mul(scaled_learning_rate)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                            torch.clone(p_grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(p_grad, alpha=1 - dampening)
                    if nesterov:
                        p_grad = p_grad.add(buf, alpha=momentum)
                    else:
                        p_grad = buf
                p.add_(-p_grad)

        return loss

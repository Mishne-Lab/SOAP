import torch
import torch.nn as nn
import torch.optim as optim

from math import exp
from functools import partial

from cw_attack import L2Adversary
from df_attack import DeepFool
from spm_attack import spm

# from defense import purify
    
G, G0 = [], []

def purify(model, aux_criterion, X, epsilon=8/255, alpha=4/255, num_iter=5):
    
    # epsilon, alpha, num_iter = 0.3, 0.1, 5
    aux_track = torch.zeros(11, X.shape[0])
    inv_track = torch.zeros(11, *X.shape)
    for e in range(11):
        defense = partial(defense_wrapper, criterion=aux_criterion, defense='pgd_linf', epsilon=e*epsilon/2, alpha=alpha, num_iter=num_iter)
        inv_delta = defense(model, X=X)
        inv_track[e] = inv_delta
        aux_track[e, :] = aux_criterion(model, (X+inv_delta).clamp(0,1)).detach()
    e_selected = aux_track.argmin(dim=0)
    return inv_track[e_selected, torch.arange(X.shape[0])].to(X.device) + X

def empty(model, criterion, X, y=None, epsilon=0.1, bound=(0,1)):
    return torch.zeros_like(X)

def inject_noise(X, epsilon=0.1, bound=(0,1)):
    """ Construct FGSM adversarial examples on the examples X"""
    # model.eval()
    return (X + torch.randn_like(X) * epsilon).clamp(*bound) - X

def fgsm(model, criterion, X, y=None, epsilon=0.1, bound=(0,1)):
    """ Construct FGSM adversarial examples on the examples X"""
    # model.eval()
    delta = torch.zeros_like(X, requires_grad=True)
    if y is None:
        loss = criterion(model, X + delta)
    else:
        loss = criterion(model(X + delta), y)
    loss.backward()
    if y is None:
        delta = epsilon * delta.grad.detach().sign()
    else:
        delta = epsilon * delta.grad.detach().sign()
    return (X + delta).clamp(*bound) - X

def pgd_linf(model, criterion, X, y=None, epsilon=0.1, bound=(0,1), alpha=0.01, num_iter=40, randomize=False):
    """ Construct PGD adversarial examples on the examples X"""
    # model.eval()
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    g = []
    for t in range(num_iter):
        if y is None:
            loss = criterion(model, X + delta)
        else:
            loss = criterion(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = (X + delta).clamp(*bound) - X
        delta.grad.zero_()

    delta.data = (X + delta).clamp(*bound) - X
    return delta.detach()

def bpda(model, criterion, X, y=None, epsilon=0.1, bound=(0,1), alpha=0.01, num_iter=40, purify=purify):

    delta = torch.zeros_like(X)
    for t in range(num_iter):

        X_pfy = purify(model, X=X + delta).detach()
        X_pfy.requires_grad_()

        loss = criterion(model(X_pfy), y)
        loss.backward()

        delta.data = (delta + alpha*X_pfy.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = (X + delta).clamp(*bound) - X
        X_pfy.grad.zero_()

    return delta.detach()

def cw(model, criterion, X, y=None, epsilon=0.1, num_classes=10):
    delta = L2Adversary()(model, X.clone().detach(), y, num_classes=num_classes).to(X.device) - X
    delta_norm = torch.norm(delta, p=2, dim=(1,2,3), keepdim=True) + 1e-4
    delta_proj = (delta_norm > epsilon) * delta / delta_norm * epsilon + (delta_norm < epsilon) * delta
    return delta_proj

def df(model, criterion, X, y=None, epsilon=0.1):
    delta = DeepFool()(model, X.clone().detach()).clamp(0,1) - X
    delta_norm = torch.norm(delta, p=2, dim=(1,2,3), keepdim=True)
    delta_proj = (delta_norm > epsilon) * delta / delta_norm * epsilon + (delta_norm < epsilon) * delta
    return delta_proj

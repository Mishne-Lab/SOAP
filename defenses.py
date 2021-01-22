import torch
import torch.nn as nn
import torch.optim as optim

from functools import partial

from attacks import *

def defense_wrapper(model, criterion, X, defense, epsilon=None, alpha=None, num_iter=None):
    
    model.aux = True
    if defense == 'fgsm':
        inv_delta = fgsm(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon)
    elif defense == 'pgd_linf':
        inv_delta = pgd_linf(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
    elif defense == 'inject_noise':
        inv_delta = inject_noise(X, epsilon)
    model.aux = False
    # model.eval()
    return inv_delta

def purify(model, aux_criterion, X, defense='pgd_linf', epsilon=8/255, alpha=4/255, num_iter=5):

    aux_track = torch.zeros(11, X.shape[0])
    inv_track = torch.zeros(11, *X.shape)
    for e in range(11):
        defense = partial(defense_wrapper, criterion=aux_criterion, defense=defense, epsilon=e*epsilon/2, alpha=alpha, num_iter=num_iter)
        inv_delta = defense(model, X=X)
        aux_track[e, :] = aux_criterion(model, (X+inv_delta).clamp(0,1))
    e_selected = aux_track.argmin(dim=1)
    return inv_track(e_selected, torch.arange(X.shape[0])) + X
import torch
import torch.nn as nn
import torch.optim as optim

from functools import partial

from attacks import fgsm, pgd_linf, inject_noise

def defense_wrapper(model, criterion, X, defense, epsilon=None, step_size=None, num_iter=None):
    
    model.aux = True
    if defense == 'fgsm':
        inv_delta = fgsm(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon)
    elif defense == 'pgd_linf':
        inv_delta = pgd_linf(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon, step_size=step_size, num_iter=num_iter)
    elif defense == 'inject_noise':
        inv_delta = inject_noise(X, epsilon)
    else:
        raise TypeError("Unrecognized defense name: {}".format(defense))
    model.aux = False
    # model.eval()
    return inv_delta

def purify(model, aux_criterion, X, defense_mode='pgd_linf', delta=4/255, step_size=4/255, num_iter=5):

    if aux_criterion is None:
        return X
    aux_track = torch.zeros(11, X.shape[0])
    inv_track = torch.zeros(11, *X.shape)
    for e in range(11):
        defense = partial(defense_wrapper, criterion=aux_criterion, defense=defense_mode, epsilon=e*delta, step_size=step_size, num_iter=num_iter)
        inv_delta = defense(model, X=X)
        inv_track[e] = inv_delta
        aux_track[e, :] = aux_criterion(model, (X+inv_delta).clamp(0,1)).detach()
    e_selected = aux_track.argmin(dim=0)
    return inv_track[e_selected, torch.arange(X.shape[0])].to(X.device) + X
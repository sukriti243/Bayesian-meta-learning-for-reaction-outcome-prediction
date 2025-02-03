import torch
import numpy as np

def hypergradient(query_loss, support_loss, params_outer, params_inner):
    
    v = torch.autograd.grad(query_loss, params_inner, retain_graph=True)
    dtrain_dinner = torch.autograd.grad(support_loss, params_inner, retain_graph=True, create_graph=True)
    p = approxInverseHVP(v, dtrain_dinner, params_inner)
    g_indirect = torch.autograd.grad(dtrain_dinner, params_outer, grad_outputs=p, retain_graph=True)
    return list(-g for g in g_indirect)

def approxInverseHVP(v, dtrain_dinner, params_inner):
    p = v  
    alpha = 1e-4
    p_vec = cat_list_to_tensor(p)
    
    for j in range(5):
        # p_prev_vec = p_vec
        grad = torch.autograd.grad(dtrain_dinner, params_inner, grad_outputs=v, retain_graph=True)
        v = [v_ - alpha * g for v_, g in zip(v, grad)]
        p = [p_ + v_ for p_, v_ in zip(p, v)]

    return [alpha*pp for pp in p]

def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])

def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.prior_params(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()

def mix_grad(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad
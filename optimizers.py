__author__ = 'hiroki'

from collections import OrderedDict

import theano.tensor as T

from utils import build_shared_zeros


def main(name, cost, params, emb, x, w=None, lr=0.1):
    if name == 'adagrad':
        assert w is not None
        return ada_grad(cost=cost, params=params, emb=emb, x=x, w=w, lr=lr)
    else:
        return sgd(cost=cost, params=params, emb=emb, x=x, lr=lr)


def sgd(cost, params, emb, x, lr=0.1):
    updates = OrderedDict()
    grads = T.grad(cost, params)

    """update sub-tensor of embeddings"""
    updates[emb] = T.inc_subtensor(x, -lr * T.grad(cost, x))

    """update parameters"""
    for p, g in zip(params, grads):
        updates[p] = p - lr * g
    return updates


def ada_grad(cost, params, emb, x, w, lr=0.1, eps=1.):
    updates = OrderedDict()
    grads = T.grad(cost, params)

    """update sub-tensor of embeddings"""
    p = emb
    g = T.grad(cost, x)
    r = build_shared_zeros(p.get_value(True).shape)
    r_sub = r[w]
    r_sub_t = r_sub + T.sqr(g)
    r_t = T.set_subtensor(r_sub, r_sub_t)
    p_t = T.inc_subtensor(x, - (lr / (T.sqrt(r_sub_t) + eps)) * g)
    updates[r] = r_t
    updates[p] = p_t

    """update parameters"""
    for p, g in zip(params, grads):
        r = build_shared_zeros(p.get_value(True).shape)
        r_t = r + T.sqr(g)
        p_t = p - (lr / (T.sqrt(r_t) + eps)) * g
        updates[r] = r_t
        updates[p] = p_t
    return updates


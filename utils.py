#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np


def score_normalization(ref, com, cohorts, top=-1):
    """
    Adaptive symmetric score normalization using cohorts from eval data
    """
    def ZT_norm(ref, com, top=-1):
        """
        Perform Z-norm or T-norm depending on input order
        """
        S = np.mean(np.inner(cohorts, ref), axis=1)
        S = np.sort(S, axis=0)[::-1][:top]
        mean_S = np.mean(S)
        std_S = np.std(S)
        score = np.inner(ref, com)
        score = np.mean(score)
        return (score - mean_S) / std_S

    def S_norm(ref, com, top=-1):
        """
        Perform S-norm
        """
        return (ZT_norm(ref, com, top=top) + ZT_norm(com, ref, top=top)) / 2

    ref = ref.cpu().numpy()
    com = com.cpu().numpy()
    return S_norm(ref, com, top=top)

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)
"""
Activation aparsity class to create, propagate, and freeze binary masks during training.

Copyright (c) Deeplite Inc.
"""

import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
from .scheduler import *

def _lnorm_masking(x, sparsity, score):
    shape = x.shape
    nparams = round(sparsity * shape[2] * shape[3]) # toprune
    mask = torch.ones(1, shape[2], shape[3]).to(x.device)
    if nparams:
        score2d = torch.norm(score, p=1, dim=0)
        topk = torch.topk(score2d.view(-1), k=nparams, largest=False)
        mask[0, :, :].view(-1)[topk.indices] = 0
    return torch.tile(mask, dims=(shape[1], 1, 1))

def _random_masking(x, sparsity):
    shape = x.shape
    nparams = round(sparsity * shape[2] * shape[3]) # toprune
    mask = torch.ones(1, shape[2], shape[3]).to(x.device)
    idx = torch.rand(shape[2], shape[3])
    topk = torch.topk(idx.view(-1), k=nparams)
    mask[0, :, :].view(-1)[topk.indices] = 0
    return torch.tile(mask, dims=(shape[1], 1, 1))

class APruner(ABC):
    def __init__(self, modules, config, device):
        self.named_modules = modules
        self.scheduler = PolyDecayScheduler(
                begin_step=config['begin_step'],
                end_step=config['end_step'],
                frequency=config['frequency'],
                dtype=float,
                initial_value=config['init_value'], # same for each layer
                final_value=config['final_value'],
                power=2,
        )
        self.shapes = config['shapes']
        self.imgsz = config['imgsz']
        self.criterion = config['criterion']
        if self.criterion == 'random':
            self.prune_foo = _random_masking
        elif self.criterion == 'lnorm':
            self.prune_foo = _lnorm_masking
        else:
            raise NotImplementedError
        self.device = device
        self.sparsity = 0
        self.freeze_step = config['freeze_step']
        self.frozen_mask = False
        self._register_pruner()
        self.l0_score = torch.rand(1, self.imgsz, self.imgsz).to(self.device)
        self.best_l0_score = self.l0_score

    def _register_pruner(self):
        for name, module in self.named_modules.items():
            shape = self.shapes[name]
            module.register_buffer(
                'mask', torch.ones(shape[1], shape[2], shape[3]).to(self.device),
            )
            module.sparsity = .0
            module.frozen = False # if True no mask update
            module.make_mask = self.prune_foo
            def masking(module, i, o):
                if module.sparsity:
                    if module.frozen is not True:
                        module.mask = module.make_mask(o, module.sparsity, module.score)
                    return o.mul_(torch.tile(module.mask, dims=(o.shape[0], 1, 1, 1)))
                return o
            module.register_forward_hook(masking)
        return

    def step(self, global_step):
        if not self.frozen_mask and (global_step >= self.freeze_step):
            self.propagate_score(self.best_l0_score)
            self.freeze_mask()
        if self.scheduler.should_do(global_step) is not True: # dense step
            if self.sparsity: # coming from pruning step
                self._reset_mask()
            return False
        else: # sparse step
            self._prune(self.scheduler.get_value(global_step))
            if self.frozen_mask is not True:
                self.propagate_score(
                    torch.rand(1, self.imgsz, self.imgsz).to(self.device)
                )
            return True

    def hard_step(self, global_step): # force pruning
        self._prune(self.scheduler.get_value(global_step))
        return

    def _prune(self, sparsity): # induce target sparsity level
        self.sparsity = sparsity
        for name, module in self.named_modules.items():
            module.sparsity = self.sparsity
        return

    def _reset_mask(self):
        self.sparsity = .0
        for name, module in self.named_modules.items():
            module.sparsity = self.sparsity
        return

    def freeze_mask(self):
        for name, module in self.named_modules.items():
            module.sparsity = self.sparsity
            o = torch.randn(self.shapes[name], device=self.device)
            module.mask = module.make_mask(o, module.sparsity, module.score)
            module.frozen = True
        self.frozen_mask = True
        print(" Pruning Masks Frozen!")
        return

    def propagate_score(self, l0_score):
        self.l0_score = l0_score
        l0_res = l0_score.shape[-1]
        for name, module in self.named_modules.items():
            _, _, oh, ow = self.shapes[name]
            ds_ratio = l0_res // oh
            module.score = torch.nn.functional.avg_pool2d(l0_score, ds_ratio)
            module.score.requires_grad = False
        return

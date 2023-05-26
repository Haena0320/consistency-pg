"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

A prefetch loader to speedup data loading
Modified from Nvidia Deep Learning Examples
(https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch).
"""
import random

import torch
from torch.utils.data import DataLoader

from utils.distributed import any_broadcast


# class MetaLoader(object):
#     """ wraps multiple data loaders """
#     def __init__(self, loaders, accum_steps=1, distributed=False):
#         assert isinstance(loaders, dict)
#         self.name2loader = {}
#         self.name2iter = {}
#         self.sampling_pools = []
#         for n, l in loaders.items():
#             if isinstance(l, tuple):
#                 l, r = l
#             elif isinstance(l, DataLoader):
#                 r = 1
#             else:
#                 raise ValueError()
#             self.name2loader[n] = l
#             self.name2iter[n] = iter(l)
#             self.sampling_pools.extend([n]*r)
#
#         self.accum_steps = accum_steps
#         self.distributed = distributed
#         self.step = 0
#
#     def __iter__(self):
#         """ this iterator will run indefinitely """
#         task = self.sampling_pools[0]
#         while True:
#             if self.step % self.accum_steps == 0:
#                 task = random.choice(self.sampling_pools)
#                 if self.distributed:
#                     # make sure all process is training same task
#                     task = any_broadcast(task, 0)
#             self.step += 1
#             iter_ = self.name2iter[task]
#             try:
#                 batch = next(iter_)
#             except StopIteration:
#                 iter_ = iter(self.name2loader[task])
#                 batch = next(iter_)
#                 self.name2iter[task] = iter_
#
#             yield task, batch


class MetaLoader(object):
    """ wraps multiple data loaders """
    def __init__(self, loaders, accum_steps=1, distributed=False):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        self.ratio_ = []
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1

            else:
                raise ValueError()

            self.name2loader[n] = l
            self.name2iter[n] = iter(l)

            if not n.startswith("vcr"):
                self.sampling_pools.extend([n]*r)

            self.ratio_.append((n, r))
        print(f"batch sampling ratio initialized as {self.ratio_}")

        self.accum_steps = accum_steps
        self.distributed = distributed
        self.step = 0
        self.t0 = 20000
        self.k = 0.1

    def __iter__(self):
        """ this iterator will run indefinitely """

        while True:
            task = random.choice(self.sampling_pools)

            if self.step % self.accum_steps == 0 :
                if self.step > self.t0:
                    self._update_ratio()
                print(self.sampling_pools)

            if self.distributed:
                # make sure all process is training same task
                task = any_broadcast(task, 0)

            self.step += 1
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_
            yield task, batch

    def _update_ratio(self):
        lambda_ = self.k*(self.step-self.t0)
        lambda_ = torch.sigmoid(torch.tensor(lambda_))
        print(f'batch sampling ratio is convered into lambda_ : {lambda_} ')
        de_lambda_ = torch.tensor(1) - lambda_

        # if self.step > self.t0 * 0.75:
        #     pass
        # else:
        new_sampling_pools = []
        for i, (n, r) in enumerate(self.ratio_):
            print(n,r)
            if n.startswith("vcr") :
                new_r = r * (lambda_)
            elif n.startswith("itm") :
                #new_r = max(r * de_lambda_, torch.tensor(1).float())
                new_r = r * de_lambda_
            elif n.startswith("mlm"):
                new_r = r * de_lambda_
                #new_r = max(r * de_lambda_, torch.tensor(1).float())
            else:
                raise ValueError
            print(f'batch sampling ratio of {n} task is {new_r}')
            new_sampling_pools.extend([n] * torch.round(new_r).long().item())
        self.sampling_pools = new_sampling_pools


def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method

# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------

import torch.utils.data as data
import torch.distributed as dist
import pytorch_lightning as pl
from .mugen_data import MUGENDataset

class VideoData(pl.LightningDataModule):

    def __init__(self, args, shuffle=True):
        super().__init__()
        self.args = args
        self.shuffle = shuffle

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def _dataset(self, split):

        Dataset = MUGENDataset(args=self.args, split=split)
        return Dataset

    def _dataloader(self, split):
        dataset = self._dataset(split)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None and self.shuffle is True,
            collate_fn=None
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader('train')

    def val_dataloader(self):
        return self._dataloader('val')

    def test_dataloader(self):
        return self._dataloader('test')
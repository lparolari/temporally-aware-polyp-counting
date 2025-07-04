import pathlib
import random

import lightning as L
import torch
import torch.utils

from polypsense.dataset import InstanceDataset
from polypsense.dataset.repo.fragmenter import Fragmenter
from polypsense.dataset.repo.identifier import Identifier
from polypsense.dataset.repo.trackleter import Trackleter
from polypsense.e2e.data import FragmentIdentityDataset
from polypsense.e2e.sampler import MultiPosConBatchSampler
from polypsense.e2e.transforms import (
    affine,
    anchor_crop,
    color_jitter,
    compose,
    hflip,
    random_anchor_crop,
    random_iou_crop,
    to_tensor,
    vflip,
)


class End2EndDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_root,
        *,
        im_size,
        fragment_length,
        fragment_stride,
        fragment_drop_last,
        fragment_padding_mode,
        bbox_scale_factor,
        bbox_scale_range,
        min_tracklet_length,
        aug_vflip,
        aug_hflip,
        aug_affine,
        aug_colorjitter,
        aug_scalebboxes,
        aug_ioucrop,
        batch_size,
        num_workers,
        n_views,
        seed,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.im_size = im_size
        self.fragment_length = fragment_length
        self.fragment_stride = fragment_stride
        self.fragment_drop_last = fragment_drop_last
        self.fragment_padding_mode = fragment_padding_mode
        self.bbox_scale_factor = bbox_scale_factor
        self.bbox_scale_range = bbox_scale_range
        self.min_tracklet_length = min_tracklet_length
        self.aug_vflip = aug_vflip
        self.aug_hflip = aug_hflip
        self.aug_affine = aug_affine
        self.aug_colorjitter = aug_colorjitter
        self.aug_scalebboxes = aug_scalebboxes
        self.aug_ioucrop = aug_ioucrop
        self.batch_size = batch_size
        self.eval_n_views = 2
        self.eval_batch_size = 6
        self.num_workers = num_workers
        self.n_views = n_views
        self.seed = seed
        self.rng = random.Random(seed)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_ds("train")
            self.val_dataset = self._make_ds("val")

        if stage == "test" or stage is None:
            self.test_dataset = self._make_ds("test")

    def train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._get_eval_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def _get_train_dataloader(self, ds):
        batch_sampler = MultiPosConBatchSampler(
            ds.y,
            k=self.n_views,
            batch_size=self.batch_size,
            seed=self.rng.randint(0, 2**32 - 1),
        )
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def _get_eval_dataloader(self, ds):
        batch_sampler = MultiPosConBatchSampler(
            ds.y,
            k=self.eval_n_views,
            batch_size=self.eval_batch_size,
            seed=self.seed,
        )
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def _make_ds(self, split):
        dataset_root = pathlib.Path(self.dataset_root)
        img_folder = dataset_root / "images"
        ann_file = dataset_root / "annotations" / f"instances_{split}.json"

        instance_ds = InstanceDataset.from_instances(img_folder, ann_file)

        trackleter = Trackleter(instance_ds)
        identifier = Identifier(instance_ds)
        fragmenter = Fragmenter(
            trackleter,
            min_tracklet_length=self.min_tracklet_length,
            fragment_length=self.fragment_length,
            stride=self.fragment_stride,
            drop_last=self.fragment_drop_last,
            padding_mode=self.fragment_padding_mode,
        )

        ds = FragmentIdentityDataset(
            instance_ds,
            fragmenter,
            identifier,
            frame_transforms=self._get_frame_transforms(split),
            sequence_transforms=self._get_sequence_transforms(split),
            return_dict=True,
        )

        return ds

    def _get_frame_transforms(self, split):
        return {
            "train": compose(to_tensor()),
            "val": compose(to_tensor()),
            "test": compose(to_tensor()),
        }[split]

    def _get_sequence_transforms(self, split):
        use_anchor_crop = lambda: (
            random_anchor_crop(*self.bbox_scale_range, self.im_size)
            if self.aug_scalebboxes
            else anchor_crop(self.bbox_scale_factor, self.im_size)
        )

        return {
            "train": compose(
                use_anchor_crop()
                + random_iou_crop(self.aug_ioucrop, self.im_size)
                + hflip(self.aug_hflip)
                + vflip(self.aug_vflip)
                + affine(self.aug_affine)
                + color_jitter(self.aug_colorjitter)
            ),
            "val": compose(anchor_crop(self.bbox_scale_factor, self.im_size)),
            "test": compose(anchor_crop(self.bbox_scale_factor, self.im_size)),
        }[split]


def collate_fn(batch):
    return torch.utils.data.default_collate(batch)

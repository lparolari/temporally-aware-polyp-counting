import torch

from polypsense.dataset import CocoDataset


class InstanceDataset(torch.utils.data.Dataset):
    def __init__(self, ds: CocoDataset):
        self.ds = ds
        self._build_ids()

    def _build_ids(self):
        self.targets = [
            t for i in range(len(self.ds)) for t in self.ds.tgt_ann(self.ds.id(i))
        ]
        self.ids = [i for i in range(len(self.targets))]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.id(index)
        return self.img(id), self.tgt_ann(id)

    def id(self, index):
        return self.ids[index]

    def img(self, id):
        target = self.targets[id]
        image_id = target["image_id"]
        return self.ds.img(image_id)

    def img_ann(self, id):
        target = self.targets[id]
        image_id = target["image_id"]
        return self.ds.img_ann(image_id)

    def tgt_ann(self, id):
        return self.targets[id]

    @staticmethod
    def from_instances(root, ann_file):
        return InstanceDataset(CocoDataset(root, ann_file))

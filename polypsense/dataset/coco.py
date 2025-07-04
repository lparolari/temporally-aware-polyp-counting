import contextlib
import os

import torchvision


class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(
        self, root, annFile, transform=None, target_transform=None, transforms=None
    ):
        # suppress pycocotools prints
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                super().__init__(root, annFile, transform, target_transform, transforms)

    def id(self, index):
        return self.ids[index]

    def img(self, id):
        return self._load_image(id)

    def img_ann(self, id):
        return self.coco.loadImgs(id)[0]

    def tgt_ann(self, id):
        return self._load_target(id)

    def from_instances(root, ann_file):
        return CocoDataset(root, ann_file)

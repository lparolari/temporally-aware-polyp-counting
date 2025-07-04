import torch
from torchvision.tv_tensors import Video, BoundingBoxes


from polypsense.dataset.repo.fragmenter import Fragmenter
from polypsense.dataset.repo.identifier import Identifier


class FragmentIdentityDataset:
    """
    A dataset of pairs (fragment, identity).
    """

    def __init__(
        self,
        instance_ds,
        fragmenter,
        identifier,
        frame_transforms=None,
        sequence_transforms=None,
        return_dict=False,
    ):
        self.instance_ds = instance_ds
        self.fragmenter: Fragmenter = fragmenter
        self.identifier: Identifier = identifier
        self.frame_transforms = frame_transforms or (lambda x: x)
        self.sequence_transforms = sequence_transforms or (lambda x: x)
        self.return_dict = return_dict

        self.ids: list[int] = None
        self.x: list = None
        self.y: list = None
        self._build()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        fragment, label = self.x[id], self.y[id]

        frames = self._get_frames(fragment)  # [s, c, w, h]
        bboxes = self._get_bboxes(fragment)  # [s, 4]

        out = self.sequence_transforms({"frames": frames, "bboxes": bboxes})  # [s, c, w, h]  # fmt: skip

        clip = out["frames"]

        if self.return_dict:
            return {
                "clip": clip,
                "bboxes": out["bboxes"],
                "label": label,
                "position": self._get_position(fragment),
            }

        return clip, label  # [s, c, w, h], [1]

    def _get_frames(self, fragment):
        # TODO: here we use `frame_transforms` just to convert PIL.Image to
        # tensor, we should remove `frame_transforms` and use
        # `sequence_transforms` as `transforms`
        frames = [self.instance_ds.img(frame_id) for frame_id in fragment]
        frames = [self.frame_transforms(frame) for frame in frames]
        return Video(torch.stack(frames))

    def _get_bboxes(self, fragment):
        img_ann_0 = self.instance_ds.img_ann(fragment[0])

        format = "xywh"
        canvas_size = (img_ann_0["height"], img_ann_0["width"])

        bboxes = [self.instance_ds.tgt_ann(frame_id)["bbox"] for frame_id in fragment]
        bboxes = torch.tensor(bboxes)
        bboxes = BoundingBoxes(bboxes, format=format, canvas_size=canvas_size)
        return bboxes

    def _get_position(self, fragment):
        """
        Computes the position of the current fragment wrt the identity. That is,
        the position is the normalized distance of the fragment from the start
        of the identity.
        """
        instances = self.identifier.get_instances(
            self.identifier.find_identity_id(fragment[0].item())
        )
        first, last = instances[0], instances[-1]

        first_frame_id = self.instance_ds.img_ann(first)["frame_id"]
        last_frame_id = self.instance_ds.img_ann(last)["frame_id"]

        identity_length = last_frame_id - first_frame_id

        current_frame_id = self.instance_ds.img_ann(fragment[0])["frame_id"]

        pos = current_frame_id - first_frame_id
        pos_norm = pos / identity_length

        return pos_norm

    def _build(self):
        """
        Construct the list of fragments and labels, where labels have been
        mapped from identities to integers.
        """
        self.x = torch.tensor(self.fragmenter.fragments())
        self.ids = list(range(len(self.x)))

        identities = [
            self.identifier.find_identity_id(self.fragmenter.fragment(fragment_id)[0])
            for fragment_id in self.ids
        ]
        identities_unique = sorted(list(set(identities)))

        self.identity2label = {
            identity: idx for idx, identity in enumerate(identities_unique)
        }
        self.label2identity = {
            idx: identity for identity, idx in self.identity2label.items()
        }

        self.y = torch.tensor(
            [self.identity2label[identity] for identity in identities]
        )

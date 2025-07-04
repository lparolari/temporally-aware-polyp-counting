import numpy as np


class Trackleter:
    """
    Build a list of tracklets out of a collection of items. Tracklets are
    separated based on temporal and spatial constraints.
    """

    def __init__(
        self,
        ds,
        separation_thresh=1,
        iou_thresh=0.1,
        identity_key="identity_id",
        pbar=False,
    ):
        self.ds = ds
        self.separation_thresh = separation_thresh
        self.iou_thresh = iou_thresh
        self.identity_key = identity_key
        self.pbar = pbar

        self._build_tracklets()

    def _build_tracklets(self):
        # 1. get data for each instance for easier manipulation
        # 2. sort by (identity_id, frame_id) -- this is important because we can
        #    speed-up the retrieval of the maximum frame_id just by accessing
        #    the last item in the tracklet with `tracklet[-1]`
        # 3. build tracklets iteratively:
        #    - for each identity_id, take consecutive instances, namely a and b
        #    - if abs(frame_id_b - frame_id_a) <= separation_thresh and iou(b,
        #      a) >= iou_thresh then add b to the tracklet
        #    - otherwise, start a new tracklet

        # ********** 1 **********
        identity_ids = []
        frame_ids = []
        bboxes = []
        id2idx = {}
        idx2id = {}

        for i in range(len(self.ds)):
            identity_ids.append(self.ds.tgt_ann(self.ds.id(i))[self.identity_key])
            frame_ids.append(self.ds.img_ann(self.ds.id(i))["frame_id"])
            bboxes.append(self.ds.tgt_ann(self.ds.id(i))["bbox"])
            id2idx[self.ds.id(i)] = i
            idx2id[i] = self.ds.id(i)

        # ********** 2 **********
        identity_ids = np.array(identity_ids)
        frame_ids = np.array(frame_ids)
        bboxes = np.array(bboxes)

        sort_idx = np.lexsort((frame_ids, identity_ids))

        # ********** 3 **********
        identities = np.unique(identity_ids).tolist()
        tracklets = []

        for identity_id in identities:

            tracklet = []

            for j in range(len(sort_idx)):
                i = sort_idx[j]
                instance_id = self.ds.id(i)

                assert i == id2idx[instance_id]
                assert instance_id == idx2id[i]

                # skip instances that do not belong to the current identity
                if identity_ids[id2idx[instance_id]] != identity_id:
                    continue

                # when found, initialize the tracklet if first instance
                if len(tracklet) == 0:
                    tracklet.append(instance_id)
                # or add to the tracklet if the conditions are met
                else:
                    is_sequential = (
                        abs(frame_ids[id2idx[instance_id]] - frame_ids[id2idx[tracklet[-1]]])
                        <= self.separation_thresh
                    )
                    is_overlapping = (
                        iou(
                            xywh2xyxy(bboxes[id2idx[instance_id]]),
                            xywh2xyxy(bboxes[id2idx[tracklet[-1]]]),
                        )
                        >= self.iou_thresh
                    )
                    if is_sequential and is_overlapping:
                        tracklet.append(instance_id)
                    else:
                        tracklets.append(tracklet)
                        tracklet = [instance_id]
            # end for i

            tracklets.append(tracklet)

        self._tracklets = tracklets
        self._ids = list(range(len(self._tracklets)))

    def __getitem__(self, idx):
        return self._tracklets[self._ids[idx]]

    def __len__(self):
        return len(self._ids)


def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou(a_bbox, b_bbox):
    a_x1, a_y1, a_x2, a_y2 = a_bbox
    b_x1, b_y1, b_x2, b_y2 = b_bbox

    # get the overlap rectangle
    x1 = max(a_x1, b_x1)
    y1 = max(a_y1, b_y1)
    x2 = min(a_x2, b_x2)
    y2 = min(a_y2, b_y2)

    # check if there is an overlap
    if x2 < x1 or y2 < y1:
        return 0

    # if there is an overlap, calculate the area
    intersection_area = (x2 - x1) * (y2 - y1)

    # calculate the area of both boxes
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)

    # calculate the union area
    union_area = a_area + b_area - intersection_area

    return intersection_area / union_area

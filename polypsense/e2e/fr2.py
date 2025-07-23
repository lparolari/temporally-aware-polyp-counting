import argparse
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

from polypsense.e2e.dm import End2EndDataModule
from polypsense.e2e.clustering import associate, get_clustering


def run(
    exp_name: str,
    dataset_root: str,
    split: str,
    output_path: str,
    encoder_type: str,
    encoder_ckpt: str,
    clustering_type: str,
    clustering_hparams: dict | None,
    im_size: int,
    fragment_length: int,
    fragment_stride: int = 4,
    fragment_drop_last: bool = False,
    target_fpr: float = 0.05,
    average: str = "micro",
    bbox_scale_factor: float = 1.0,
):
    wandb.init(
        name=exp_name,
        entity="lparolari",
        project="polypsense-fr",
        dir=os.path.join(os.getcwd(), "wandb_logs"),
        config={
            "dataset_root": dataset_root,
            "split": split,
            "encoder_type": encoder_type,
            "encoder_ckpt": encoder_ckpt,
            "clustering_type": clustering_type,
            "clustering_hparams": clustering_hparams,
            "target_fpr": target_fpr,
            "average": average,
            "im_size": im_size,
            "fragment_length": fragment_length,
            "fragment_stride": fragment_stride,
            "fragment_drop_last": fragment_drop_last,
            "bbox_scale_factor": bbox_scale_factor,
        },
    )

    run_id = wandb.run.id
    print("run_id", run_id)

    dataset_root = pathlib.Path(dataset_root)
    output_dir = pathlib.Path(output_path) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = get_encoder(encoder_type, encoder_ckpt)
    clustering = get_clustering(clustering_type, clustering_hparams)

    tps_list = []
    fps_list = []
    fns_list = []
    tns_list = []
    recalls_list = []
    precisions_list = []
    fprs_list = []
    tprs_list = []
    preds_list = []
    tracklets_list = []
    entities_list = []

    ds = get_ds(
        dataset_root=dataset_root,
        split=split,
        im_size=im_size,
        fragment_length=fragment_length,
        fragment_stride=fragment_stride,
        fragment_drop_last=fragment_drop_last,
        bbox_scale_factor=bbox_scale_factor,
    )

    for i in tqdm(range(len(ds))):
        x, y, ann = ds[i]
        video_name = ann["video_name"]

        samples_features = get_features(encoder, x)
        samples_labels = y

        n = len(y)
        n_entities = len(samples_labels.unique())

        tracklets_list.append(n)
        entities_list.append(n_entities)

        print("n =", n)
        print("n_entities =", n_entities)

        targets = get_targets(samples_labels)
        scores = get_scores(samples_features)

        print(targets)
        print(scores)

        show_heatmap(scores, output_dir / f"scores_{video_name}.png")
        show_heatmap(targets, output_dir / f"targets_{video_name}.png")

        parameters_space = clustering.parametrize()
        p = len(parameters_space)

        # feats may have other channels
        feats = torch.stack(
            [
                scores,
            ],
            dim=0,
        )  # [c, n, n]

        preds = torch.stack(
            [
                clustering.fit_predict(feats, parameters)
                for parameters in parameters_space
            ]
        )

        print(preds[0])
        show_heatmap(preds[0], output_dir / f"preds_0_{video_name}.png")

        preds_list.append(preds)

        targets = targets.reshape(-1, n, n)  # [1, g, g]

        tps = ((preds == 1) & (targets == 1)).float().view(p, -1).sum(-1)  # [p]
        fps = ((preds == 1) & (targets == 0)).float().view(p, -1).sum(-1)  # [p]
        fns = ((preds == 0) & (targets == 1)).float().view(p, -1).sum(-1)  # [p]
        tns = ((preds == 0) & (targets == 0)).float().view(p, -1).sum(-1)  # [p]

        if len(y.unique()) >= 2:
            tps_list.append(tps)
            fps_list.append(fps)
            fns_list.append(fns)
            tns_list.append(tns)

            recalls = tps / (tps + fns)
            precisions = tps / (tps + fps)
            fprs = fps / (fps + tns)  # https://en.wikipedia.org/wiki/False_positive_rate  # fmt: skip
            fprs = fprs.nan_to_num(nan=0.0)  # fixes the case when tps + fns = 0 (i.e. single polyp video)  # fmt: skip
            tprs = recalls

            recalls_list.append(recalls)
            precisions_list.append(precisions)
            fprs_list.append(fprs)
            tprs_list.append(tprs)

            print(f"{video_name} recalls", tps / (tps + fns))
            print(f"{video_name} precisions", tps / (tps + fps))
            print(f"{video_name} fprs", fps / (fps + tns))

        print(f"{video_name} initial fr", n / n_entities)
        print(f"{video_name} fr preds0", len(associate(preds[0])) / n_entities)

        wandb.log(
            {
                "scores": wandb.Image(str(output_dir / f"scores_{video_name}.png")),
                "targets": wandb.Image(str(output_dir / f"targets_{video_name}.png")),
                "preds_0": wandb.Image(str(output_dir / f"preds_0_{video_name}.png")),
            }
        )

    if average == "micro":
        tps = torch.stack(tps_list).sum(0)  # [v, p] -> [p]
        fps = torch.stack(fps_list).sum(0)  # [v, p] -> [p]
        fns = torch.stack(fns_list).sum(0)  # [v, p] -> [p]
        tns = torch.stack(tns_list).sum(0)  # [v, p] -> [p]

        recalls = tps / (tps + fns)
        precisions = tps / (tps + fps)
        fprs = fps / (fps + tns)  # https://en.wikipedia.org/wiki/False_positive_rate
        tprs = recalls

    if average == "macro":
        recalls = torch.stack(recalls_list).mean(0)
        precisions = torch.stack(precisions_list).mean(0)
        fprs = torch.stack(fprs_list).mean(0)
        tprs = torch.stack(tprs_list).mean(0)

    def sort_and_interpolate(x, y):
        sorted_indices = torch.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        x_interp = np.linspace(0, 1, num=1000)
        y_interp = np.interp(x_interp, x_sorted.numpy(), y_sorted.numpy())

        return torch.tensor(x_interp), torch.tensor(y_interp)

    def auc(x, y):
        return torch.trapz(y, x)

    fprs_roc, tprs_roc = sort_and_interpolate(fprs, tprs)
    recalls_roc, precisions_roc = sort_and_interpolate(recalls, precisions)

    auroc = auc(fprs_roc, tprs_roc)
    auprc = auc(recalls_roc, precisions_roc)

    show_p_r(recalls_roc, precisions_roc, output_dir / "p_r_sorted.png")
    show_roc_curve(fprs_roc, tprs_roc, output_dir / "roc_curve.png")
    show_pr_curve(recalls_roc, precisions_roc, output_dir / "pr_curve.png")

    print("auroc", auroc)
    print("auprc", auprc)

    best_i = select_best_parameters(fprs, target_fpr=target_fpr)
    best_parameters = clustering.parametrize()[best_i]
    operating_fpr = fprs[best_i]

    show_p_r(recalls, precisions, output_dir / "p_r.png", best_i=best_i)

    print("best_i", best_i)
    print("best_parameters", best_parameters)
    print("operating_fpr", operating_fpr)
    print("precision@best_i", precisions[best_i])
    print("recall@best_i", recalls[best_i])

    show_heatmap(preds[best_i], output_dir / f"preds_best_{video_name}.png")

    preds_per_video = [preds[best_i] for preds in preds_list]
    fragments_per_video = [associate(pred) for pred in preds_per_video]

    fr_per_video = [
        len(f) / n_entities for f, n_entities in zip(fragments_per_video, entities_list)
    ]
    initial_fr_per_video = [
        n / n_entities for n, n_entities in zip(tracklets_list, entities_list)
    ]

    fr = torch.tensor(fr_per_video).mean()
    fr_std = torch.tensor(fr_per_video).std()
    initial_fr = torch.tensor(initial_fr_per_video).mean()

    print("initial_fr_per_video", initial_fr_per_video)
    print("fr_per_video", fr_per_video)
    print("initial fr", initial_fr)
    print("fr", fr)
    print("fr_std", fr_std)
    print("run_id", run_id)

    wandb.log(
        {
            "fr": fr,
            "fr_std": fr_std,
            "auroc": auroc.item(),
            "auprc": auprc.item(),
            "fpr": operating_fpr.item(),
            "best_parameters": best_parameters,
            "p_r": wandb.Image(str(output_dir / "p_r.png")),
            "roc_curve": wandb.Image(str(output_dir / "roc_curve.png")),
            "pr_curve": wandb.Image(str(output_dir / "pr_curve.png")),
        }
    )


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------


import re


class ReidDataset:
    def __init__(self, ds, x, y, labels):
        self.ds = ds
        self.x = x
        self.y = y

        # parse video id from labels (works just for realcolon)
        pattern = r"(\d+-\d+)_\d+(\.\d+)?"

        self.videos_labels = [re.match(pattern, label).group(1) for label in labels]
        self.unique_videos = sorted(list(set(self.videos_labels)))

    def __getitem__(self, idx):
        vid = self.unique_videos[idx]
        idxs = [i for i, item in enumerate(self.videos_labels) if item == vid]

        clips = []
        labels = []
        for idx in idxs:
            fragment = self.x[idx]
            label = self.y[idx]
            ann = {"video_name": vid}

            frames = self.ds._get_frames(fragment)  # list of PIL images
            bboxes = self.ds._get_bboxes(fragment)

            out = self.ds.sequence_transforms({"frames": frames, "bboxes": bboxes})  # [s, c, w, h]  # fmt: skip

            # clip = torch.stack([ds.frame_transforms(d)["frames"] for d in data])  # [s, c, w, h]  # fmt: skip
            # clip = ds.sequence_transforms(clip)  # [s, c, w, h]
            # clips.append(clip)

            clip = out["frames"]
            clips.append(clip)

            labels.append(label)

        clip = torch.stack(clips)
        labels = torch.tensor(labels)

        return clip, labels, ann  # [s, c, w, h], [1]

    def __len__(self):
        return len(self.unique_videos)


def get_ds(
    dataset_root,
    split,
    im_size,
    fragment_length,
    fragment_stride,
    fragment_drop_last,
    bbox_scale_factor,
):
    dm = End2EndDataModule(
        dataset_root,
        im_size=im_size,
        fragment_length=fragment_length,
        fragment_stride=fragment_stride,
        fragment_drop_last=fragment_drop_last,
        fragment_padding_mode=None if fragment_drop_last else "repeat",
        bbox_scale_factor=bbox_scale_factor,
        min_tracklet_length=30,
        aug_vflip=False,
        aug_hflip=False,
        aug_affine=False,
        aug_colorjitter=False,
        aug_scalebboxes=False,
        aug_ioucrop=False,
        num_workers=6,
        # sampler related params
        batch_size=None,
        n_views=None,
        seed=None,
        bbox_scale_range=None,
        bbox_diagonal_ratio_range=None,
    )

    if split == "val":
        ds = dm._make_ds("val")
    elif split == "test":
        ds = dm._make_ds("test")
    else:
        raise ValueError(f"Unknown split {split}")

    return ReidDataset(
        ds, ds.x, ds.y, labels=[ds.label2identity[label.item()] for label in ds.y]
    )


# -----------------------------------------------------------------------------
# MODEL SUPPORT
# -----------------------------------------------------------------------------


def get_encoder(encoder_type, encoder_ckpt):
    if encoder_type == "mve":
        from polypsense.e2e.model import MultiViewEncoder

        # base_ckpt = MultiViewEncoder.load_from_checkpoint("/home/lparolar/Projects/polypsense/wandb_logs/polypsense-mve_tmp/ymortr7r/checkpoints/epoch=25-step=3588-val_loss=1.96.ckpt")
        # # create a temporary ckpt which is identical to encoder_ckpt, the only difference is that we copy the state_dict of sfe model into that
        # ckpt = torch.load(encoder_ckpt)
        # ckpt["hyper_parameters"]["sfe"] = base_ckpt.hparams["sfe"]
        # # save to tmp.ckpt
        # tmp_ckpt = "tmp.ckpt"
        # torch.save(ckpt, tmp_ckpt)
        model = MultiViewEncoder.load_from_checkpoint(encoder_ckpt, map_location="cpu")
        model.eval()
        model.cuda()

        return model
    elif encoder_type == "sfe":
        from polypsense.e2e.model import SingleFrameEncoder

        # base_ckpt = SingleFrameEncoder.load_from_checkpoint("/home/lparolar/Projects/polypsense/wandb_logs/polypsense-mve_tmp/z395g428/checkpoints/epoch=2-step=435-val_loss=1.66.ckpt")
        # ckpt = torch.load(encoder_ckpt)
        # ckpt["hyper_parameters"]["backbone"] = base_ckpt.hparams["backbone"]
        # # # save to tmp.ckpt
        # tmp_ckpt = "tmp.ckpt"
        # torch.save(ckpt, tmp_ckpt)
        model = SingleFrameEncoder.load_from_checkpoint(
            encoder_ckpt, map_location="cpu"
        )
        model.eval()
        model.cuda()

        return model
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")


def get_features(encoder, x):
    # x [n, s, 3, 232, 232]
    embs = []
    for fragment in x:
        with torch.no_grad():
            fragment = fragment.cuda()  # [s, 3, 232, 232]
            fragment = fragment.unsqueeze(0)  # [1, s, 3, 232, 232]
            emb = encoder(fragment)  # [1, 128]
            emb = emb.squeeze(0).cpu()  # [128]
            embs.append(emb)
    return torch.stack(embs)  # [n, 128]


def get_scores(x):
    """
    Return similarity scores between all pairs of samples.

    Args:
        x: [n, d] features

    Returns:
        scores: [n, n] similarity matrix
    """
    # get pairwise distances
    dists = torch.cdist(x, x, p=2)

    # normalize matrix
    # dists = (dists - dists.mean()) / dists.std()
    d_min = dists.min()
    d_max = dists.max()
    dists = (dists - d_min) / (d_max - d_min)

    # the bigger the better
    scores = 1 - dists  # [g, g]

    return scores


def get_targets(y):
    """
    Return a boolean matrix indicating whether pairs of samples have the same
    label.

    Args:
        y: [n] labels

    Returns:
        targets: [n, n] boolean matrix
    """
    return y.unsqueeze(1) == y.unsqueeze(0)


def select_best_parameters(fprs, target_fpr):
    """
    Return the index of value that minimizes the distance with the target value.
    In case of multiple matches, the median index is returned for stability.
    """
    fprs = fprs.masked_fill(fprs == 0, float("inf"))
    diff = torch.abs(fprs - target_fpr)
    min_value = torch.min(diff)
    min_indices = torch.nonzero(diff == min_value, as_tuple=True)[0]
    median_index = min_indices.median().item()
    return median_index


# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------


def show_heatmap(x, out_path):
    plt.imshow(x, cmap="viridis")
    plt.savefig(out_path)
    plt.close()


def show_p_r(recalls, precisions, out_path, best_i=None):
    x = torch.arange(recalls.size(0))
    plt.plot(x, recalls, label="Recall")
    plt.plot(x, precisions, label="Precision")
    if best_i is not None:
        plt.axvline(best_i, color="red", linestyle="--", label="Best FPR")
    plt.xlabel("Param group")
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def show_roc_curve(fprs, tprs, out_path):
    plt.plot(fprs, tprs)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(out_path)
    plt.close()


def show_pr_curve(recalls, precisions, out_path):
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(out_path)
    plt.close()


def auc(x, y, reorder=False):
    # Source: https://pytorch.org/torcheval/main/generated/torcheval.metrics.functional.auc.html

    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor([])

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)

    if reorder:
        x, x_idx = torch.sort(x, dim=1, stable=True)
        y = y.gather(1, x_idx)

    return torch.trapz(y, x)


# -----------------------------------------------------------------------------
# RUNTIME
# -----------------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--output_path", type=str, default="output", required=True)
    parser.add_argument(
        "--encoder_type",
        type=str,
        required=True,
        choices=["mve", "sfe"],
    )
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument(
        "--clustering_type",
        type=str,
        required=True,
        choices=[
            "threshold",
            "affinity_propagation",
            "temporal_affinity_propagation",
        ],
    )
    parser.add_argument("--clustering_hparams", type=json.loads, default=None)
    parser.add_argument("--target_fpr", type=float, default=0.05)
    parser.add_argument(
        "--metric_average", type=str, default="micro", choices=["micro", "macro"]
    )
    parser.add_argument("--im_size", type=int, default=232)
    parser.add_argument("--fragment_length", type=int, default=8)
    parser.add_argument("--fragment_stride", type=int, default=4)
    parser.add_argument("--fragment_drop_last", action="store_true", default=False)
    parser.add_argument("--bbox_scale_factor", type=float, default=1.0)

    return parser


def main():
    args = get_parser().parse_args()

    run(
        exp_name=args.exp_name,
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        encoder_type=args.encoder_type,
        encoder_ckpt=args.encoder_ckpt,
        clustering_type=args.clustering_type,
        clustering_hparams=args.clustering_hparams,
        split=args.split,
        target_fpr=args.target_fpr,
        average=args.metric_average,
        fragment_length=args.fragment_length,
        fragment_stride=args.fragment_stride,
        fragment_drop_last=args.fragment_drop_last,
        bbox_scale_factor=args.bbox_scale_factor,
        im_size=args.im_size,
    )


if __name__ == "__main__":
    main()

import argparse

from polypsense.e2e.engine import eval, train


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.mode and args.mode == "eval":
        eval(args)
    else:
        train(args)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str)

    # training
    parser.add_argument("--mode", type=str, choices=["train", "eval"])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # datamodule
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--fragment_length", type=int)
    parser.add_argument("--fragment_stride", type=int)
    parser.add_argument("--fragment_drop_last", action="store_true")
    parser.add_argument("--fragment_padding_mode", type=str)
    parser.add_argument("--bbox_scale_factor", type=float)
    parser.add_argument("--bbox_scale_range", type=float, nargs=2)
    parser.add_argument("--min_tracklet_length", type=int)
    parser.add_argument("--im_size", type=int)
    parser.add_argument("--n_views", type=int)

    # augmentation
    parser.add_argument("--aug_vflip", action="store_true")
    parser.add_argument("--aug_hflip", action="store_true")
    parser.add_argument("--aug_affine", action="store_true")
    parser.add_argument("--aug_colorjitter", action="store_true")
    parser.add_argument("--aug_scalebboxes", action="store_true")
    parser.add_argument("--aug_ioucrop", action="store_true")

    # model
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--backbone_arch", type=str, choices=["resnet18", "resnet50"])  # fmt: skip
    parser.add_argument("--backbone_weights", choices=["IMAGENET1K_V1", "IMAGENET1K_V2"])  # fmt: skip

    # experiment
    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--exp_project", type=str)

    return parser


if __name__ == "__main__":
    main()

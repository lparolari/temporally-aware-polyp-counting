import os

import torch
import lightning as L

from polypsense.e2e.dm import End2EndDataModule
from polypsense.e2e.model import MultiViewEncoder


def train(args):
    L.seed_everything(args.seed, workers=True)

    dm = get_datamodule(args)
    model = get_model(args)

    trainer = L.Trainer(
        logger=get_logger(args, model),
        callbacks=get_callbacks(args),
        max_epochs=args.max_epochs,
        accelerator="cuda",
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.test(model, dm)
    trainer.fit(model, dm, ckpt_path=args.resume and args.ckpt_path)
    trainer.test(model, dm, ckpt_path="best")


def eval(args):
    L.seed_everything(args.seed, workers=True)

    dm = get_datamodule(args)
    model = get_model(args)

    trainer = L.Trainer(
        logger=get_logger(args, model),
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
        accelerator="cuda",
        strategy="auto",
        devices=1,
        num_nodes=1,
    )

    trainer.test(model, dm)


def get_model(args):
    model = MultiViewEncoder(
        backbone_arch=args.backbone_arch,
        backbone_weights=args.backbone_weights,
        d_proj=128,
        d_model=args.d_model,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        lr=args.lr,
        temperature=args.temperature,
    )

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")

        missing_keys, unexpected_keys = model.load_state_dict(ckpt["state_dict"])

        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    return model


def get_datamodule(args):
    return End2EndDataModule(
        dataset_root=args.dataset_root,
        im_size=args.im_size,
        fragment_length=args.fragment_length,
        fragment_stride=args.fragment_stride,
        fragment_drop_last=args.fragment_drop_last,
        fragment_padding_mode=args.fragment_padding_mode,
        bbox_scale_factor=args.bbox_scale_factor,
        bbox_scale_range=args.bbox_scale_range,
        min_tracklet_length=args.min_tracklet_length,
        aug_vflip=args.aug_vflip,
        aug_hflip=args.aug_hflip,
        aug_affine=args.aug_affine,
        aug_colorjitter=args.aug_colorjitter,
        aug_scalebboxes=args.aug_scalebboxes,
        aug_ioucrop=args.aug_ioucrop,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_views=args.n_views,
        seed=args.seed,
    )


def get_logger(args, model):
    logger = L.pytorch.loggers.WandbLogger(
        project=args.exp_project,
        id=args.exp_id,
        name=args.exp_name,
        save_dir=os.path.join(os.getcwd(), "wandb_logs"),
        allow_val_change=True,
        resume="allow",
        config=vars(args),
    )
    # wandb.watch breaks the ability to save nn.Modules as hparams. This happens
    # because the watch method attach a non-pickble object to the module and
    # Pytorch Lightning removes non pickable objects from the hparams before
    # saving it. See https://github.com/wandb/wandb/issues/2588.
    return logger


def get_callbacks(args):
    return [
        # save best ckpt
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{step}-{val_loss:.2f}",
        ),
        # save last ckpt (for resuming)
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="step",
            mode="max",
            filename="{epoch}-{step}",
        ),
        L.pytorch.callbacks.LearningRateMonitor(),
    ]

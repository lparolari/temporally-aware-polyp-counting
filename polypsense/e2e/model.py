import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DistanceAwareMultiPosConLoss(nn.Module):
    """
    Adaptation of Multi-Positive Contrastive Loss
    (https://arxiv.org/pdf/2306.00984.pdf) to support soft targets originated
    from distances between samples.
    """

    def __init__(self, temperature=0.1, lam=1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.lam = lam
        self.return_logits = return_logits

    def forward(self, feats, labels, positions):
        # feats: [b, d]
        # labels: [b]

        b = feats.size(0)
        device = feats.device

        feats = F.normalize(feats, dim=-1, p=2)  # [b, d]

        targets = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()  # [b, b]

        # get the self mask, note that original implementation uses
        # `torch.scatter` to mask portion of the matrix interesed by current
        # node (rank) computations:
        # https://github.com/google-research/syn-rep-learn/blob/45f451b0d53d25eecdb4d7b9e5a852e1c43e7f5b/StableRep/models/losses.py#L78-L91
        mask = 1 - torch.eye(b).to(device)  # [b] (float)

        # apply self-masking on targets
        targets = targets * mask  # [b, b]

        # compute logits and apply self masking on logits, -1e9 is used to get 0
        # probability in cross-entry
        logits = torch.matmul(feats, feats.T) / self.temperature  # [b, b]

        # apply self-masking on logits: logits' maximum value is 1 on the
        # diagonal (from matmul of normalized feat), so
        # - on the diagonal remove (1 - mask) * 1e9, where mask is 0, i.e. -1e9
        # - off the diag remove (1 - mask) * 1e9, where mask is 1, i.e. 0
        logits = logits - (1 - mask) * 1e9  # [b, b]

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)  # [b, b]

        # compute supervised ground-truth distribution
        p = targets / targets.sum(1, keepdim=True).clamp(min=1.0)  # [b, b]

        # compute soft targets with distance awareness
        distances = torch.abs(positions.view(-1, 1) - positions.view(1, -1))  # [b, b]
        distances = torch.exp(-self.lam * distances)  # [b, b]
        distances = distances * targets  # [b, b]
        distances = distances / distances.sum(1, keepdim=True).clamp(
            min=1e-8
        )  # to distribution

        # apply distance-awareness on supervised targets
        p = p * distances
        p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # compute cross-entropy loss
        loss = compute_cross_entropy(p, logits)

        if self.return_logits:
            return loss, logits

        return loss


def compute_cross_entropy(p, q):
    """
    See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    for details.

    NOTE: this should return the same value as `F.cross_entropy` but for some
    reason original authors reimplemented it. The signature is different though:
    p comes first.
    """
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return -loss.mean()


def stablize_logits(logits):
    """
    Stabilizes the input logits by subtracting the maximum value along the last
    dimension.

    This technique is often used to improve numerical stability when computing
    functions involving exponentials, such as the softmax function. By shifting
    the logits, it prevents potential overflow or underflow issues.
    """
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


def accuracy(logits, labels, topk=(1,)):
    """
    For each row in the logits matrix (query item), find the column (gallery item)
    with the maximum logit value (excluding the diagonal if it's self-similarity).
    Check if the predicted maximum corresponds to a ground truth positive label.

    logits: [b, b]
    labels: [b]
    """
    with torch.no_grad():
        device = logits.device

        # Create a binary mask for positive pairs
        targets = torch.eq(labels.view(-1, 1), labels.view(1, -1)).to(device)  # [b, b]

        # Mask diagonal (self-similarities): we don't want to count a match
        # between query and itself in gallery. For this reason we mask the
        # diagonal.
        n = logits.size(0)
        mask = ~torch.eye(n, dtype=bool).to(device)
        logits = logits.masked_fill(~mask, float("-inf"))

        # Get top-k predictions as an index tensor where for each row (query) we
        # have the top-k indices of the gallery items.
        maxk = max(topk)
        _, preds = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [b, k]

        # Given that preds is an index to the top-k prediction for each query,
        # we construct a matrix such that for each i (query) we have k j
        # (predictions) where corrects[i, j] is True if the j-th prediction is
        # positive with respect to the query. Thus, correct [i, 0] is the top-1
        # prediction, if it is True then the top-1 prediction is positive for i.
        corrects = targets.gather(1, preds)  # [b, k]

        # Check if ground truth is in top-K
        accuracies = []
        for k in topk:
            corrects_k = corrects[:, :k].any(-1)  # [b]
            correct_k = corrects_k.float().sum()  # [1]
            accuracies.append(correct_k * 100.0 / n)
        return accuracies


class MultiViewEncoder(L.LightningModule):
    """
    Implements the Multi-View Encoder model as described in (Intrator et al.
    2023). Implementation detals follow (Chen et al. 2020).

    - z is the projected representation used for contrastive learning during training
    - h is the representation used for downstream tasks

    ### Architecture

    ```
        z                              128-dim
        |
       PROJ     g(h)
       HEAD
        | h                            2048-dim
    +---------------------------+
    |    Transformer Encoder    |
    +---------------------------+
        |       |        |
        |      SFE      SFE            2048-dim
        |       |        |
        CLS   frame1   frame2
    ```
    """

    def __init__(
        self,
        backbone_arch="resnet50",
        backbone_weights="IMAGENET1K_V2",
        d_proj=128,
        d_model=2048,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        lr=1e-4,
        temperature=0.07,
    ):
        super().__init__()

        self.sfe = SingleFrameEncoder(backbone_arch, backbone_weights)
        self.lr = lr
        self.temperature = temperature

        if d_model != self.sfe.out_features:
            # project sfe output to d_model
            self.adapter = nn.Linear(self.sfe.out_features, d_model)

        self.transformer_encoder = self._build_transformer_encoder(
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # [1, 1, d]

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )

        self.criterion = DistanceAwareMultiPosConLoss(
            temperature=self.temperature, lam=1, return_logits=True
        )

        # https://github.com/Lightning-AI/pytorch-lightning/issues/11494#issuecomment-1235866844
        self.save_hyperparameters(ignore=["sfe"])
        self.save_hyperparameters(logger=False)

    def forward(self, x):
        # x: [b, s, c, h, w]
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)  # [b*s, c, h, w]
        x = self.sfe(x)  # [b*s, d]
        if hasattr(self, "adapter"):
            x = self.adapter(x)
        x = x.view(b, s, -1)  # [b, s, d]
        x = self._prepend_cls_token(x)  # [b, 1+s, d]
        x = x.permute(1, 0, 2)  # [1+s, b, d]
        x = self.transformer_encoder(x)  # [s+1, b, d]
        return x[0]  # [b, d]

    def step(self, batch, stage=None):
        x, y = batch["clip"], batch["label"]
        position = batch["position"]
        h = self(x)
        z = self.head(h)  # [b, p]
        loss, logits = self.criterion(z, y, position)  # [1], [b, b]

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            (top1, top5) = accuracy(logits, y, topk=(1, 5))
            self.log(f"{stage}_acc_top1", top1)
            self.log(f"{stage}_acc_top5", top5)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        return optimizer

    def _prepend_cls_token(self, x):
        # x: [b, s, d]
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)  # [b, 1, d]
        return torch.cat((cls_tokens, x), dim=1)  # [b, 1+s, d]

    def _build_transformer_encoder(
        self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout
    ):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
        )
        return transformer_encoder


class SingleFrameEncoder(nn.Module):
    def __init__(self, backbone_arch, backbone_weights):
        super().__init__()

        self.backbone = ResNet(backbone_arch, backbone_weights)
        self.out_features = self.backbone.out_features

        # add mlp projection head
        d_feats = self.out_features
        d_head = 128  # d_head is not hparam relevent to us
        self.head = nn.Sequential(
            nn.Linear(d_feats, d_feats),
            nn.ReLU(),
            nn.Linear(d_feats, d_head),
        )

    def forward(self, images):
        return self.backbone(images)


class ResNet(nn.Module):

    def __init__(self, base_model, pretrained):
        super().__init__()

        self.resnet = self._get_basemodel(base_model, pretrained)

        # save the feature dimension with a similar api as `nn.Linear`
        self.out_features = self.resnet.fc.in_features

        # then, remove the fully connected layer: we want to use the unprojected
        # representation
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)

    def _get_basemodel(self, model_name, pretrained):
        weights = pretrained or None
        resnet_dict = {
            "resnet18": lambda: models.resnet18(weights=weights),
            "resnet50": lambda: models.resnet50(weights=weights),
        }

        if model_name not in resnet_dict:
            raise ValueError(
                f"Invalid '{model_name}' backbone. Use one of: {list(resnet_dict.keys())}"
            )

        return resnet_dict[model_name]()

import torch


class MultiPosConBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels, *, k=2, batch_size=8, seed=42):
        """
        Builds batches of `batch_size` items, where `k` items belong to the same
        class. A class may appear multiple time in a batch, but there must be at
        least two different classes in batch. Maximum number of different
        classes in batch is `m = batch_size // k` classes.
        """
        if batch_size % k != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by k ({k})")

        self.rng = torch.Generator().manual_seed(seed)
        self.labels = (
            labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
        )
        self.k = k
        self.batch_size = batch_size
        self.indices = self._build_indices()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def _build_indices(self):
        indices = []

        labels = self.labels
        labels_unique = labels.unique()
        label2idx = {c: torch.where(labels == c)[0] for c in labels_unique.tolist()}

        n = len(labels)
        b = self.batch_size
        k = self.k
        m = b // k

        while n > 0:
            # - sample m unique labels
            # - for each label, sample k data points
            # - remove sampled data points from available data points
            # - update n

            available = torch.tensor([len(v) >= k for v in label2idx.values()])
            availability = available.sum().item()  # number of available classes for this batch  # fmt: skip
            numerosity = torch.tensor([len(v) for v in label2idx.values()])

            # this potentially discards a lot of data...
            if availability < m:
                break

            weights = numerosity * available
            weights = weights / weights.sum()

            index_sampled = torch.multinomial(
                weights,
                m,
                replacement=False,
                generator=self.rng,
            )
            labels_sampled = labels_unique[index_sampled]

            # get k samples from each label, then remove samples from available,
            # update n
            batch = []

            for label in labels_sampled:
                idx = label2idx[label.item()]
                idx = idx[torch.randperm(len(idx), generator=self.rng)]
                idx1 = idx[:k]
                idx2 = idx[k:]

                # remove idx from available
                label2idx[label.item()] = idx2

                batch.extend(idx1.tolist())

                n -= len(idx1)

            if len(batch) == self.batch_size:
                indices.append(batch)

        indices = torch.tensor(indices)
        # shuffle indices inside each batch to avoid having consecutive positive
        # samples
        indices = indices[:, torch.randperm(b, generator=self.rng)]

        return indices

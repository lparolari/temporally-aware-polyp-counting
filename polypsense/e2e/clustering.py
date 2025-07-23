import torch

Clusters = list[list[int]]


def associate(preds: torch.Tensor) -> Clusters:
    """
    Return fragments given a boolean matrix indicating whether pairs of samples
    belong to the same group.

    Args:
        preds: [n, n] boolean matrix

    Returns:
        fragments: list of list of int
    """
    n = preds.size(0)

    # Initialize groups
    merged_groups = []
    tracklet_to_group = {}

    preds = {(i, j): preds[i, j].item() for i in range(n) for j in range(n)}

    # Iterate through all tracklet pairs
    for (t1, t2), pred in preds.items():
        if pred:
            # If both tracklets are not in any group, create a new group
            if t1 not in tracklet_to_group and t2 not in tracklet_to_group:
                new_group = {t1, t2}
                merged_groups.append(new_group)
                tracklet_to_group[t1] = new_group
                tracklet_to_group[t2] = new_group
            # If one of the tracklets is already in a group, add the other to the group
            elif t1 in tracklet_to_group and t2 not in tracklet_to_group:
                group = tracklet_to_group[t1]
                group.add(t2)
                tracklet_to_group[t2] = group
            elif t2 in tracklet_to_group and t1 not in tracklet_to_group:
                group = tracklet_to_group[t2]
                group.add(t1)
                tracklet_to_group[t1] = group
            # If both tracklets are in different groups, merge the two groups
            elif (
                t1 in tracklet_to_group
                and t2 in tracklet_to_group
                and tracklet_to_group[t1] != tracklet_to_group[t2]
            ):
                group1 = tracklet_to_group[t1]
                group2 = tracklet_to_group[t2]
                merged_groups.remove(group2)  # Remove the smaller group
                group1.update(group2)  # Merge both groups
                for t in group2:
                    tracklet_to_group[t] = (
                        group1  # Update all tracklets to point to the new merged group
                    )

    # Add tracklets that were not merged into their own groups
    for tracklet in list(range(n)):
        if tracklet not in tracklet_to_group:
            merged_groups.append({tracklet})

    return [list(g) for g in merged_groups]


def get_clustering(clustering_type: str, parameters=None):
    clustering_cls = {
        "threshold": ThresholdClustering,
        "affinity_propagation": AffinityPropagationClustering,
        "temporal_affinity_propagation": TemporalAffinityPropagationClustering,
    }[clustering_type]

    return clustering_cls(parameters)


class Clustering:
    def parametrize(self) -> list:
        """
        Return a list of parameters to be tested.

        Returns:
            parameters: list of dict
        """
        pass

    def fit_predict(self, feats: torch.Tensor, parameters: dict) -> torch.Tensor:
        """
        Args:
            feats: [c, n, n] matrix of features with c channels where the first
              one is [n, n] similarity matrix, the second is [n, n] gaps matrix
            parameters: dict of algorithm related parameters

        Returns:
            preds: [n, n] boolean matrix
        """
        pass


class ThresholdClustering(Clustering):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters

    def parametrize(self):
        if self.parameters:
            return [self.parameters]
        thresholds = torch.linspace(0, 1, 101).tolist()
        return [{"threshold": t} for t in thresholds]

    def fit_predict(self, feats, parameters):
        scores = feats[0]
        thresh = parameters["threshold"]

        clusters = associate(scores >= thresh)

        return _clusters2preds(clusters, n=scores.size(0), device=scores.device)


class AffinityPropagationClustering(Clustering):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters

    def parametrize(self):
        if self.parameters:
            return [self.parameters]

        return [
            {
                "damping": 0.9,  # the algorithm is more stable
                "max_iter": 500,
                "convergence_iter": 50,
                "preference": preference,
                "affinity": "precomputed",
                "random_state": 42,
            }
            for preference in torch.linspace(-5, 5, 21).tolist()
        ]

    def fit_predict(self, feats, parameters):
        from sklearn.cluster import AffinityPropagation

        clustering = AffinityPropagation(**parameters)

        scores = feats[0]

        preds = torch.tensor(clustering.fit_predict(scores))
        preds = preds.unsqueeze(0) == preds.unsqueeze(1)

        return preds


class TemporalAffinityPropagationClustering(Clustering):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters

    def parametrize(self):
        if self.parameters:
            if not isinstance(self.parameters, list):
                return [self.parameters]
            return self.parameters

        alphas = torch.linspace(0.3, 0.7, 41).tolist()

        return [
            {
                "damping": 0.9,  # the algorithm is more stable
                "max_iter": 500,
                "convergence_iter": 50,
                "preference": 0,
                "affinity": "precomputed",
                "random_state": 42,
                "lambda": 1,
                "alpha": alpha,
                "fn": "sum",
            }
            for alpha in alphas
        ]

    def fit_predict(self, feats, parameters):
        from sklearn.cluster import AffinityPropagation

        lam = parameters["lambda"]
        fn = parameters["fn"]

        v = feats[0]
        distances = feats[1]
        d = torch.exp(-lam * distances)  # [b, b]
        v_norm = (v - v.min()) / (v.max() - v.min())
        d_norm = (d - d.min()) / (d.max() - d.min())

        if fn == "sum":
            alpha = parameters["alpha"]
            scores = alpha * v_norm + (1 - alpha) * d_norm

        elif fn == "mult":
            scores = v_norm * d_norm

        else:
            raise ValueError(f"Unknown function: {fn}")

        clustering = AffinityPropagation(
            damping=parameters["damping"],
            max_iter=parameters["max_iter"],
            convergence_iter=parameters["convergence_iter"],
            preference=parameters["preference"],
            affinity="precomputed",
            random_state=parameters["random_state"],
        )

        preds = torch.tensor(clustering.fit_predict(scores))
        preds = preds.unsqueeze(0) == preds.unsqueeze(1)

        return preds


def _clusters2preds(clusters: Clusters, n: int, device: str):
    preds = torch.zeros(n, n)
    for c in clusters:
        for i in c:
            for j in c:
                preds[i, j] = 1
    return preds.bool().to(device)

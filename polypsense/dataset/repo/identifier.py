IdentityId = str
VideoId = str
InstanceId = int

Instances = list[InstanceId]


class Identifier:
    """
    Build identities from instances.

    Uses the value of the `identity_key` to construct identities. If the
    `identity_key` is unset, all instance belongs to a default identity.
    """

    def __init__(
        self, ds, identity_key: str = "identity_id", default_identity="000-000_0"
    ):
        self.ds = ds
        self.identity_key = identity_key
        self.default_identity = default_identity
        self._build_identity_index()

    def get_instances(self, identity_id: IdentityId) -> Instances:
        """
        Returns a list of instance ids that belong to the given identity.
        """
        return sorted(self._identity2index[identity_id])

    def list_identity_ids(self) -> list[IdentityId]:
        """
        Returns a list of all identity ids.
        """
        return list(self._identity2index.keys())

    def find_identity_id(self, instance_id: InstanceId) -> IdentityId:
        """
        Returns the identity id for the given instance id.
        """
        return self._index2identity[instance_id]

    def get_video(self, identity_id: IdentityId) -> VideoId:
        """
        Returns the video id for the given identity id.
        """
        return identity_id.split("_")[0]

    def _build_identity_index(self):
        """
        Builds two indices:
        * identity -> list of indices, with the list of indices belonging to given identity
        * index -> identity, with the identity id for each given index
        """
        identity2index = {}
        index2identity = {}

        for i in range(len(self.ds)):
            id = self.ds.id(i)
            tgt_ann = self.ds.tgt_ann(id)

            identity_id = tgt_ann.get(self.identity_key, self.default_identity)

            if identity_id not in identity2index:
                identity2index[identity_id] = []

            identity2index[identity_id].append(id)
            index2identity[id] = identity_id

        assert len(index2identity) == len(
            [v for vs in identity2index.values() for v in vs]
        )

        self._identity2index = identity2index
        self._index2identity = index2identity

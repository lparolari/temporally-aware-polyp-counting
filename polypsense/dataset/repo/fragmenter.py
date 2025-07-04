from polypsense.dataset.repo.trackleter import Trackleter


class Fragmenter:
    """
    Builds a set of fragments from tracklets, where a fragment is a sequence
    countiguous instances from a tracklet.

    ### Example:

    given tracklets
    [0]
    [1]
    [2, 3, 4]
    [5]
    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]
    [76]
    ...
    the fragmenter will generate the following fragments:
    
    - with padding_mode="repeat", drop_last=False
    >>> self[0]
    [0, 0, 0, 0, 0, 0, 0, 0]

    - with padding_mode=None, drop_last=True
    >>> self[0]
    [6, 7, 8, 9, 10, 11, 12, 13]

    - with padding_mode=None, drop_last=True, stride=4
    >>> self[0]
    [6, 10, 14, 18, 22, 26, 30, 34]
    """

    def __init__(
        self,
        trackleter: Trackleter,
        min_tracklet_length=0,
        fragment_length=8,
        stride=1,
        drop_last=False,
        padding_mode="repeat",
    ):
        if not (fragment_length > 0 or fragment_length == -1):
            raise ValueError(
                "fragment_length must be either a positive ingeter or -1 for unbounded length"
            )

        if padding_mode and padding_mode not in ["repeat"]:
            raise ValueError(
                f"padding_mode must be one of ['repeat'], got {padding_mode}"
            )

        if not drop_last and not padding_mode:
            raise ValueError("padding_mode must be set if drop_last is False")

        if drop_last and padding_mode:
            raise ValueError("padding_mode must not be set if drop_last is True")

        self.trackleter = trackleter

        self.min_tracklet_length = min_tracklet_length
        self.fragment_length = fragment_length
        self.stride = stride
        self.drop_last = drop_last
        self.padding_mode = padding_mode
        self._build_fragments()

    def fragments(self):
        return self._fragments

    def fragment(self, id):
        return self._fragments[id]

    def _build_fragments(self):
        self._fragments = []

        for i in range(len(self.trackleter)):
            tracklet = self.trackleter[i]

            if len(tracklet) >= self.min_tracklet_length:
                self._fragments.extend(self._get_fragments_from_tracklet(tracklet))

        self._ids = list(range(len(self._fragments)))

    def _get_fragments_from_tracklet(self, tracklet):
        tracklet_stride = tracklet[:: self.stride]
        tracklet_length = len(tracklet_stride)

        if self.fragment_length == -1:
            return [tracklet_stride]

        fragments = []

        for i in range(0, tracklet_length, self.fragment_length):
            fragment = tracklet_stride[i : i + self.fragment_length]

            if len(fragment) < self.fragment_length:
                if self.drop_last:
                    continue

                padding = self.fragment_length - len(fragment)

                if self.padding_mode == "repeat":
                    fragment = fragment + [fragment[-1]] * padding

            fragments.append(fragment)

        return fragments

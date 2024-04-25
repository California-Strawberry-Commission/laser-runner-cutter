import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TrackState(Enum):
    PENDING = 1
    ACTIVE = 2
    COMPLETED = 3
    FAILED = 4


class Track:
    id: int
    pixel: Tuple[int, int]
    position: Tuple[float, float, float]
    state: TrackState

    def __init__(
        self, id: int, pixel: Tuple[int, int], position: Tuple[float, float, float]
    ):
        """
        Args:
            pixel (Tuple[int, int]): Pixel coordinates (x, y) of target in camera frame.
            position (Tuple[float, float, float]): 3D position (x, y, z) of target relative to camera.
        """
        self.id = id
        self.pixel = pixel
        self.position = position
        self.state = TrackState.PENDING


class Tracker:
    tracks: Dict[int, Track]

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Args:
            logger (Optional[logging.Logger]): Logger
        """
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)
        self.tracks = {}

    @property
    def has_pending_tracks(self) -> bool:
        return any(track.state == TrackState.PENDING for track in self.tracks.values())

    @property
    def pending_tracks(self) -> List[Track]:
        return [
            track for track in self.tracks.values() if track.state == TrackState.PENDING
        ]

    @property
    def has_active_tracks(self) -> bool:
        return any(track.state == TrackState.ACTIVE for track in self.tracks.values())

    @property
    def active_tracks(self) -> List[Track]:
        return [
            track for track in self.tracks.values() if track.state == TrackState.ACTIVE
        ]

    def add_track(
        self,
        pixel: Tuple[int, int],
        position: Tuple[float, float, float],
        track_id: Optional[int] = None,
    ):
        """Add a track to list of current tracks.

        Args:
            pixel (Tuple[int, int]): Pixel coordinates (x, y) of target in camera frame.
            position (Tuple[float, float, float]): 3D position (x, y, z) of target relative to camera.
            track_id (Optional[int]): unique instance ID assigned to the object
        """
        # TODO: If there is no assigned track ID, we may want to use additional heuristics in the future.
        # For now, just ignore it.
        if track_id is None or track_id <= 0:
            return

        # If the track already exists update that track instead of adding a new one.
        if track_id in self.tracks:
            track = self.tracks[track_id]
            track.pixel = pixel
            track.position = position
        else:
            track = Track(track_id, pixel, position)
            self.tracks[track_id] = track

    def clear(self):
        """
        Remove all tracks.
        """
        self.tracks.clear()

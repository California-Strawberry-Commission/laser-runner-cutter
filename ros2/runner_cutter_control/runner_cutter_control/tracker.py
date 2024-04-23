import logging
from enum import Enum


class TrackState(Enum):
    PENDING = 1
    ACTIVE = 2
    COMPLETED = 3
    FAILED = 4


class Track:
    def __init__(self, id, pixel, position):
        """
        Args:
            pixel (float, float): pixel coordinates (x, y) of target in camera frame
            position (float, float, float): 3D position (x, y, z) of target relative to camera
        """
        self.state = TrackState.PENDING
        self.id = id
        self.pixel = pixel
        self.position = position


class Tracker:
    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        self.tracks = {}

    @property
    def has_pending_tracks(self):
        return any(track.state == TrackState.PENDING for track in self.tracks.values())

    @property
    def pending_tracks(self):
        return [
            track for track in self.tracks.values() if track.state == TrackState.PENDING
        ]

    @property
    def has_active_tracks(self):
        return any(track.state == TrackState.ACTIVE for track in self.tracks.values())

    @property
    def active_tracks(self):
        return [
            track for track in self.tracks.values() if track.state == TrackState.ACTIVE
        ]

    def add_track(self, pixel, position, track_id=None):
        """Add a track to list of current tracks.

        Args:
            pixel (float, float): pixel coordinates (x, y) of target in camera frame
            position (float, float, float): 3D position (x, y, z) of target relative to camera
            track_id (int | None): unique instance ID assigned to the object
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
        self.tracks.clear()

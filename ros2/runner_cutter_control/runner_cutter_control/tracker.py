import logging
from enum import Enum

import numpy as np


class TrackState(Enum):
    PENDING = 1
    ACTIVE = 2
    COMPLETED = 3
    FAILED = 4


class Track:
    def __init__(self, pixel, position):
        """
        Args:
            pixel: (float, float), pixel coordinates (x, y) of target in camera frame
            position: (float, float, float), 3D position (x, y, z) of target relative to camera
        """
        self.state = TrackState.PENDING
        self.pixel = pixel
        self.position = position


class Tracker:
    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
        self.tracks = []

    @property
    def has_pending_tracks(self):
        return any(track.state == TrackState.PENDING for track in self.tracks)

    @property
    def pending_tracks(self):
        return [track for track in self.tracks if track.state == TrackState.PENDING]

    @property
    def has_active_tracks(self):
        return any(track.state == TrackState.ACTIVE for track in self.tracks)

    @property
    def active_tracks(self):
        return [track for track in self.tracks if track.state == TrackState.ACTIVE]

    def add_track(self, pixel, position, dist_threshold=0.01):
        """Add a track to list of current tracks.

        Args:
            pixel: (float, float), pixel coordinates (x, y) of target in camera frame
            position: (float, float, float), 3D position (x, y, z) of target relative to camera
            min_dist: float, distance threshold under which the target will be merged into an existing track
        """
        # If a track already exists close to the one passed in, update that track instead of
        # adding a new one.
        for track in self.tracks:
            dist = np.linalg.norm(np.array(track.position) - np.array(position))
            if dist < dist_threshold:
                track.position = position
                track.pixel = pixel
                return

        self.tracks.append(Track(pixel, position))

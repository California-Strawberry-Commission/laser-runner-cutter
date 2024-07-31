import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque


class TrackState(Enum):
    PENDING = 1  # Still needs to be burned
    ACTIVE = 2  # Actively in the process of being targeted and burned
    COMPLETED = 3  # Has successfully been burned
    OUT_OF_LASER_BOUNDS = 4  # In camera frame, but out of laser reach
    OUT_OF_FRAME = 5  # Out of camera frame


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

    def __repr__(self):
        return f"Track(id={self.id}, pixel={self.pixel}, position={self.position}, state={self.state.name})"


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
        self._pending_tracks = deque()

    def has_track_with_state(self, state: TrackState) -> bool:
        return any(track.state == state for track in self.tracks.values())

    def get_tracks_with_state(self, state: TrackState) -> List[Track]:
        return [track for track in self.tracks.values() if track.state == state]

    def get_track(self, track_id: int) -> Optional[Track]:
        """
        Get a track by ID.

        Args:
            track_id (int): Unique instance ID assigned to the object.
        Returns:
            Optional[Track]: Track with the track_id, or None if it does not exist.
        """
        return self.tracks.get(track_id, None)

    def add_track(
        self,
        track_id: int,
        pixel: Tuple[int, int],
        position: Tuple[float, float, float],
    ) -> Optional[Track]:
        """
        Add a track to list of current tracks.

        Args:
            track_id (int): Unique instance ID assigned to the object. Must be a positive integer.
            pixel (Tuple[int, int]): Pixel coordinates (x, y) of target in camera frame.
            position (Tuple[float, float, float]): 3D position (x, y, z) of target relative to camera.
        Returns:
            Optional[Track]: Track that was created or updated, or None if track was not created nor updated.
        """
        if track_id <= 0:
            return None

        # If the track already exists, update that track instead of adding a new one.
        if track_id in self.tracks:
            track = self.tracks[track_id]
            track.pixel = pixel
            track.position = position
        else:
            track = Track(track_id, pixel, position)
            self.tracks[track_id] = track
            self._pending_tracks.append(track)
        return track

    def get_next_pending_track(self) -> Optional[Track]:
        if self._pending_tracks:
            next_track = self._pending_tracks.popleft()
            next_track.state = TrackState.ACTIVE
            return next_track
        return None

    def process_track(self, track_id: int, new_state: TrackState):
        if track_id not in self.tracks:
            return

        track = self.tracks[track_id]
        if track.state == new_state:
            return

        if track.state == TrackState.PENDING:
            self._pending_tracks.remove(track)

        track.state = new_state
        if new_state == TrackState.PENDING:
            self._pending_tracks.append(track)

    def clear(self):
        """
        Remove all tracks.
        """
        self.tracks.clear()
        self._pending_tracks.clear()

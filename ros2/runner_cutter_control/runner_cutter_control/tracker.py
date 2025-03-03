from collections import Counter, deque
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from runner_cutter_control.predictor import KalmanFilterPredictor, Predictor


class TrackState(Enum):
    PENDING = auto()  # Still needs to be burned
    ACTIVE = auto()  # Actively in the process of being targeted and burned
    COMPLETED = auto()  # Has successfully been burned
    FAILED = auto()  # Failed to burn


class Track:
    id: int
    pixel: Tuple[int, int]
    position: Tuple[float, float, float]
    state_count: Dict[TrackState, int]
    predictor: Predictor

    def __init__(
        self,
        id: int,
        pixel: Tuple[int, int],
        position: Tuple[float, float, float],
        state: TrackState = TrackState.PENDING,
        predictor: Predictor = KalmanFilterPredictor(),
    ):
        """
        Args:
            id (int): Track ID.
            pixel (Tuple[int, int]): Pixel coordinates (x, y) of target in camera frame.
            position (Tuple[float, float, float]): 3D position (x, y, z) of target relative to camera.
            state (TrackState): Track state.
        """
        self.id = id
        self.pixel = pixel
        self.position = position
        self.state_count = {state: 0 for state in TrackState}
        self.state = state
        self.predictor = predictor

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state: TrackState):
        self._state = new_state
        self.state_count[new_state] += 1

    def __repr__(self):
        return f"Track(id={self.id}, pixel={self.pixel}, position={self.position}, state={self.state.name})"

    def to_dict(self):
        return {
            "id": self.id,
            "pixel": {"x": self.pixel[0], "y": self.pixel[1]},
            "position": {
                "x": self.position[0],
                "y": self.position[1],
                "z": self.position[2],
            },
            "state": self.state.name,
            "state_count": {
                state.name: count for state, count in self.state_count.items()
            },
        }


class Tracker:
    """
    Tracker maintains a collection of Tracks with state management.
    New tracks start in the PENDING state. PENDING tracks are maintained in a queue for FIFO
    ordering.
    """

    tracks: Dict[int, Track]

    def __init__(self):
        self.tracks = {}
        self._pending_tracks = deque()

    def has_track_with_state(self, state: TrackState) -> bool:
        return any(track.state == state for track in self.tracks.values())

    def get_tracks_with_state(self, state: TrackState) -> List[Track]:
        if state == TrackState.PENDING:
            return list(self._pending_tracks)

        # TODO: cache the active track(s) so we don't need to to a linear scan for it
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
        timestamp_ms: float,
        confidence: float = 1.0,
    ) -> Optional[Track]:
        """
        Add a track to list of current tracks.

        Args:
            track_id (int): Unique instance ID assigned to the object. Must be a positive integer.
            pixel (Tuple[int, int]): Pixel coordinates (x, y) of target in camera frame.
            position (Tuple[float, float, float]): 3D position (x, y, z) of target relative to camera.
            timestamp_ms (float): Timestamp, in ms, of the camera frame.
            confidence (float): Confidence score associated with the detected target.
        Returns:
            Optional[Track]: Track that was created or updated, or None if track was not created nor updated.
        """
        if track_id <= 0:
            return None

        # If the track already exists, update that track instead of adding a new one.
        # Otherwise, create a new track and set as PENDING.
        if track_id in self.tracks:
            track = self.tracks[track_id]
            track.pixel = pixel
            track.position = position
        else:
            track = Track(track_id, pixel, position, state=TrackState.PENDING)
            self.tracks[track_id] = track
            self._pending_tracks.append(track)

        # Update predictor for the track
        track.predictor.add(position, timestamp_ms, confidence)

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

    def to_dict(self):
        return {track_id: track.to_dict() for track_id, track in self.tracks.items()}

    def get_summary(self):
        state_counts = Counter(track.state for track in self.tracks.values())
        return dict(state_counts)

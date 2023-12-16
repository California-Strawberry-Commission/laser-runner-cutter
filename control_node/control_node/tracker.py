"""File: tracker.py

Keep track of objects based on observations
"""

import numpy as np
import logging


class Track:
    """Track object, represents a single object"""

    def __init__(self, pos, point):
        self.active = True
        # List of positions all associated with this object
        self.pos_wrt_cam_list = []
        self.point_list = []
        self.pos_wrt_cam_list.append(np.array(pos))
        self.point_list.append(np.array(point))
        self.corrected_laser_point = None

    @property
    def pos_wrt_cam(self):
        return np.mean(np.array(self.pos_wrt_cam_list), axis=0)

    @property
    def point(self):
        return np.mean(np.array(self.point_list), axis=0)


class Tracker:
    """Tracker object, takes in positions and associates them with specific tracks"""

    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
        self.tracks = []

    @property
    def has_active_tracks(self):
        active_tracks = self.active_tracks
        self.logger.info(
            f"Active Tracks :{len(active_tracks)} Removed_Tracks:{len(self.tracks) - len(active_tracks)}"
        )
        for track in self.tracks:
            self.logger.info(
                f"Track Pos{track.pos_wrt_cam}, Track Point{track.point}, Active:{track.active}"
            )
        for track in self.tracks:
            if track.active:
                return True
        return False

    @property
    def active_tracks(self):
        return [track for track in self.tracks if track.active]

    def add_track(self, new_pos, new_point, min_dist=0.01):
        """Add a track to list of current tracks.
        If a track already exists close to the one passed in, instead add it to that track.

        args:
            new_pos (float, float, float): new position to add to list of existing tracks
            min_dist ()

        NOTE: These tracks are currently in relation to the camera.
        """
        self.logger.info(f"Adding Track Pos:{new_pos}, Point:{new_point}")
        for track in self.tracks:
            dist = np.linalg.norm(track.pos_wrt_cam - new_pos)
            if dist < min_dist:
                track.pos_wrt_cam_list.append(new_pos)
                track.point_list.append(new_point)
                self.logger.debug(
                    f"Pos added to Track POS:{track.pos_wrt_cam}, Point:{track.point}"
                )
                return
        self.logger.info(f"New Track Pos:{new_pos}, Point:{new_point}")
        self.tracks.append(Track(new_pos, new_point))

    def deactivate(self, input_track):
        for track in self.tracks:
            if track == input_track:
                self.logger.info(f"Deactivated track pos:{track.pos_wrt_cam}")
                track.active = False

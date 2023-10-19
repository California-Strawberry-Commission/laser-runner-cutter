import ctypes
from .laser_dac import LaserDAC
import threading
import time


# Ether Dream DAC uses 16 bits (signed) for x and y
XY_BOUNDS = (-32768, 32767)

# Ether Dream DAC uses 16 bits (unsigned) for r, g, b, i
MAX_COLOR = 65535


# Define point structure for Ether Dream
class EtherDreamPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int16),
        ("y", ctypes.c_int16),
        ("r", ctypes.c_uint16),
        ("g", ctypes.c_uint16),
        ("b", ctypes.c_uint16),
        ("i", ctypes.c_uint16),
        ("u1", ctypes.c_uint16),
        ("u2", ctypes.c_uint16),
    ]


class EtherDreamError(Exception):
    """Exception used when an error is detected with EtherDream."""

    pass


class EtherDreamDAC(LaserDAC):
    """Ether Dream DAC

    Example usage:

      dac = EtherDreamDAC()
      num_connected_dacs = dac.initialize()
      dac.connect(0)

      dac.set_color(1, 0, 0, 0.1)
      dac.add_point(100, 200)
      dac.play()
      ...
      dac.clear_points()
      dac.add_point(300, 400)
      ...
      dac.stop()
      dac.close()
    """

    def __init__(self, lib_file):
        self.points = []
        self.points_lock = threading.Lock()
        self.color = (1, 1, 1, 1)  # (r, g, b, i)
        self.playing = False
        self.connected_dac_id = 0
        self.lib = ctypes.cdll.LoadLibrary(lib_file)

    def initialize(self):
        """Initialize the native library and search for online DACs"""

        print("Initializing Ether Dream DAC")
        self.lib.etherdream_lib_start()
        print("Finding available Ether Dream DACs...")

        # Ether Dream DACs broadcast once per second, so we need to wait for a bit
        # longer than that to ensure that we see broadcasts from all online DACs
        time.sleep(1.2)

        dac_count = self.lib.etherdream_dac_count()
        print(f"Found {dac_count} Ether Dream DACs")
        return dac_count

    def connect(self, dac_idx):
        print("Connecting to DAC...")
        dac_id = self.lib.etherdream_get_id(dac_idx)
        if self.lib.etherdream_connect(dac_id) < 0:
            raise EtherDreamError(f"Could not connect to DAC [{hex(dac_id)}]")
        self.connected_dac_id = dac_id
        print(f"Connected to DAC with ID: {hex(dac_id)}")

    def set_color(self, r=1, g=1, b=1, i=1):
        self.color = (r, g, b, i)

    def get_bounds(self, offset=0):
        """Return an array of points representing the corners of the outer bounds"""
        # Ether Dream DAC uses 16 bits (signed) for x and y
        return [
            (XY_BOUNDS[0] + offset, XY_BOUNDS[0] + offset),
            (XY_BOUNDS[0] + offset, XY_BOUNDS[1] - offset),
            (XY_BOUNDS[1] - offset, XY_BOUNDS[1] - offset),
            (XY_BOUNDS[1] - offset, XY_BOUNDS[0] + offset),
        ]

    def in_bounds(self, x, y):
        return (
            x >= XY_BOUNDS[0]
            and x <= XY_BOUNDS[1]
            and y >= XY_BOUNDS[0]
            and y <= XY_BOUNDS[1]
        )

    def add_point(self, x, y):
        if self.in_bounds(x, y):
            with self.points_lock:
                self.points.append((x, y))

    def remove_point(self):
        """Remove the last added point."""
        with self.points_lock:
            if self.points:
                self.points.pop()

    def clear_points(self):
        with self.points_lock:
            self.points.clear()

    def _get_frame(self, fps=30, pps=30000, transition_duration_ms=0.5):
        """Return an array of EtherDreamPoints representing the next frame that should be rendered.

        :param fps: target frames per second
        :param pps: target points per second. This should not exceed the capability of the DAC and laser projector.
        :param transition_duration_ms: duration in ms to turn the laser off between subsequent points. If we are
        rendering more than one point, we need to provide enough time between subsequent points, or else there may
        be visible streaks between the points as the galvos take time to move to the new position
        """

        # We'll use "laxel", or laser "pixel", to refer to each point that the laser projector renders, which
        # disambiguates it from "point", which refers to the (x, y) coordinates we want to have rendered

        with self.points_lock:
            # Calculate how many laxels of transition we need to add per point
            laxels_per_transition = round(transition_duration_ms / (1000 / pps))

            # Calculate how many laxels we render each point
            ppf = pps / fps
            num_points = len(self.points)
            laxels_per_point = round(ppf if num_points == 0 else ppf / num_points)
            laxels_per_frame = (
                laxels_per_point if num_points == 0 else laxels_per_point * num_points
            )

            # Prepare frame
            FrameType = EtherDreamPoint * (laxels_per_frame)
            frame = FrameType()

            if num_points == 0:
                # Even if there are no points to render, we still to send over laxels so that we don't underflow the DAC buffer
                for frameLaxelIdx in range(laxels_per_frame):
                    frame[frameLaxelIdx] = EtherDreamPoint(0, 0, 0, 0, 0, 0, 0, 0)
            else:
                for pointIdx, point in enumerate(self.points):
                    for laxelIdx in range(laxels_per_point):
                        # Pad BEFORE the "on" laxel so that the galvo settles first, and only if there is more than one point
                        isTransition = (
                            num_points > 1 and laxelIdx < laxels_per_transition
                        )
                        frameLaxelIdx = pointIdx * laxels_per_point + laxelIdx
                        frame[frameLaxelIdx] = EtherDreamPoint(
                            int(point[0]),
                            int(point[1]),
                            0 if isTransition else int(self.color[0] * MAX_COLOR),
                            0 if isTransition else int(self.color[1] * MAX_COLOR),
                            0 if isTransition else int(self.color[2] * MAX_COLOR),
                            0 if isTransition else int(self.color[3] * MAX_COLOR),
                            0,
                            0,
                        )
            return frame

    def play(self, fps=30, pps=30000, transition_duration_ms=0.5):
        """Start playback of points.
        Ether Dream max rate: 100K pps

        :param fps: target frames per second
        :param pps: target points per second. This should not exceed the capability of the DAC and laser projector.
        :param transition_duration_ms: duration in ms to turn the laser off between subsequent points. If we are
        rendering more than one point, we need to provide enough time between subsequent points, or else there may
        be visible streaks between the points as the galvos take time to move to the new position
        """

        def playback_thread():
            while self.playing:
                frame = self._get_frame(fps, pps, transition_duration_ms)

                self.lib.etherdream_wait_for_ready(self.connected_dac_id)

                self.lib.etherdream_write(
                    self.connected_dac_id,
                    ctypes.pointer(frame),
                    len(frame),
                    len(frame) * fps,
                    1,
                )
            self.lib.etherdream_stop(self.connected_dac_id)

        if not self.playing:
            self.playing = True
            self.playback_thread = threading.Thread(target=playback_thread, daemon=True)
            self.playback_thread.start()

    def stop(self):
        if self.playing:
            self.playing = False
            self.playback_thread.join()
            self.playback_thread = None

    def close(self):
        if self.connected_dac_id:
            self.lib.etherdream_stop(self.connected_dac_id)
            self.lib.etherdream_disconnect(self.connected_dac_id)

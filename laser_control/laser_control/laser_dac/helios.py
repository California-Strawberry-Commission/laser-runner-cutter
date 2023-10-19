import ctypes
from .laser_dac import LaserDAC
import threading


# Helios DAC uses 12 bits (unsigned) for x and y
XY_BOUNDS = (4095, 4095)

# Helios DAC uses 8 bits (unsigned) for r, g, b, i
MAX_COLOR = 255


# Define point structure for Helios
class HeliosPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint16),
        ("y", ctypes.c_uint16),
        ("r", ctypes.c_uint8),
        ("g", ctypes.c_uint8),
        ("b", ctypes.c_uint8),
        ("i", ctypes.c_uint8),
    ]


class HeliosDAC(LaserDAC):
    """Helios DAC

    Example usage:

      dac = HeliosDAC()
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
        self.dac_idx = 0
        self.lib = ctypes.cdll.LoadLibrary(lib_file)

    def initialize(self):
        print("Initializing Helios DAC")
        num_devices = self.lib.OpenDevices()
        print(f"Found {num_devices} Helios DACs")
        return num_devices

    def connect(self, dac_idx):
        self.dac_idx = dac_idx

    def set_color(self, r=1, g=1, b=1, i=1):
        self.color = (r, g, b, i)

    def get_bounds(self, offset=0):
        """Return an array of points representing the corners of the outer bounds"""
        # Helios DAC uses 12 bits (unsigned) for x and y
        return [
            (offset, offset),
            (offset, XY_BOUNDS[1] - offset),
            (XY_BOUNDS[0] - offset, XY_BOUNDS[1] - offset),
            (XY_BOUNDS[0] - offset, offset),
        ]

    def in_bounds(self, x, y):
        return x >= 0 and x <= XY_BOUNDS[0] and y >= 0 and y <= XY_BOUNDS[1]

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
        """Return an array of HeliosPoints representing the next frame that should be rendered.

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
            FrameType = HeliosPoint * (laxels_per_frame)
            frame = FrameType()

            if num_points == 0:
                # Even if there are no points to render, we still to send over laxels so that we don't underflow the DAC buffer
                for frameLaxelIdx in range(laxels_per_frame):
                    frame[frameLaxelIdx] = HeliosPoint(0, 0, 0, 0, 0, 0)
            else:
                for pointIdx, point in enumerate(self.points):
                    for laxelIdx in range(laxels_per_point):
                        # Pad BEFORE the "on" laxel so that the galvo settles first, and only if there is more than one point
                        isTransition = (
                            num_points > 1 and laxelIdx < laxels_per_transition
                        )
                        frameLaxelIdx = pointIdx * laxels_per_point + laxelIdx
                        frame[frameLaxelIdx] = HeliosPoint(
                            int(point[0]),
                            int(point[1]),
                            0 if isTransition else int(self.color[0] * MAX_COLOR),
                            0 if isTransition else int(self.color[1] * MAX_COLOR),
                            0 if isTransition else int(self.color[2] * MAX_COLOR),
                            0 if isTransition else int(self.color[3] * MAX_COLOR),
                        )
            return frame

    def play(self, fps=30, pps=30000, transition_duration_ms=0.5):
        """Start playback of points.
        Helios max rate: 65535 pps
        Helios max points per frame (pps/fps): 4096

        :param fps: target frames per second
        :param pps: target points per second. This should not exceed the capability of the DAC and laser projector.
        :param transition_duration_ms: duration in ms to turn the laser off between subsequent points. If we are
        rendering more than one point, we need to provide enough time between subsequent points, or else there may
        be visible streaks between the points as the galvos take time to move to the new position
        """

        def playback_thread():
            while self.playing:
                frame = self._get_frame(fps, pps, transition_duration_ms)
                statusAttempts = 0
                # Make 512 attempts for DAC status to be ready. After that, just give up and try to write the frame anyway
                while statusAttempts < 512 and self.lib.GetStatus(self.dac_idx) != 1:
                    statusAttempts += 1

                self.lib.WriteFrame(
                    self.dac_idx,
                    len(frame) * fps,
                    0,
                    ctypes.pointer(frame),
                    len(frame),
                )
            self.lib.Stop(self.dac_idx)

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
        self.lib.CloseDevices()

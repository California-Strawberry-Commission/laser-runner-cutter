import ctypes
from .laser_dac import LaserDAC
import threading
import time


# Helios DAC uses 12 bits (unsigned) for x and y
X_BOUNDS = (0, 4095)
Y_BOUNDS = (0, 4095)

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
      dac.add_point(0.1, 0.2)
      dac.play()
      ...
      dac.clear_points()
      dac.add_point(0.3, 0.4)
      ...
      dac.stop()
      dac.close()
    """

    def __init__(self, lib_file):
        self.points = []
        self.points_lock = threading.Lock()
        self.color = (1, 1, 1, 1)  # (r, g, b, i)
        self.dac_idx = -1
        self.lib = ctypes.cdll.LoadLibrary(lib_file)
        self.playing = False
        self.playback_thread = None
        self.check_connection = False
        self.check_connection_thread = None

    def initialize(self):
        print("Initializing Helios DAC")
        num_devices = self.lib.OpenDevices()
        print(f"Found {num_devices} Helios DACs")
        return num_devices

    def connect(self, dac_idx):
        self.dac_idx = dac_idx

        def check_connection_thread():
            while self.check_connection:
                if self._get_status() < 0:
                    print(f"DAC error {self._get_status()}. Attempting to reconnect.")
                    self.stop()
                    self.lib.CloseDevices()
                    self.initialize()
                time.sleep(1)

        if self.check_connection_thread is None:
            self.check_connection = True
            self.check_connection_thread = threading.Thread(
                target=check_connection_thread, daemon=True
            )
            self.check_connection_thread.start()

    def set_color(self, r=1.0, g=1.0, b=1.0, i=1.0):
        self.color = (r, g, b, i)

    def add_point(self, x, y):
        """Add a point to be rendered by the DAC.

        :param x: x coordinate normalized to [0, 1]
        :param y: y coordinate normalized to [0, 1]
        """
        if 0.0 <= x and x <= 1.0 and 0.0 <= y and y <= 1.0:
            with self.points_lock:
                self.points.append((x, y))
            return True
        else:
            return False

    def remove_point(self):
        """Remove the last added point."""
        with self.points_lock:
            if self.points:
                self.points.pop()

    def clear_points(self):
        with self.points_lock:
            self.points.clear()

    def _denormalize_point(self, x, y):
        x_denorm = round((X_BOUNDS[1] - X_BOUNDS[0]) * x + X_BOUNDS[0])
        y_denorm = round((Y_BOUNDS[1] - Y_BOUNDS[0]) * y + Y_BOUNDS[0])
        return x_denorm, y_denorm

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
                        x, y = self._denormalize_point(point[0], point[1])
                        frame[frameLaxelIdx] = HeliosPoint(
                            x,
                            y,
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
        self.stop()
        if self.check_connection:
            self.check_connection = False
            self.check_connection_thread.join()
            self.check_connection_thread = None
        self.lib.CloseDevices()

    def _get_status(self):
        # 1 means ready to receive frame
        # 0 means not ready to receive frame
        # Any negative status means error
        return self.lib.GetStatus(self.dac_idx)

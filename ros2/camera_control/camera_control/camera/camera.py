from abc import ABC, abstractmethod


class Camera(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def set_exposure(self, exposure_ms):
        pass

    @abstractmethod
    def get_frames(self):
        pass

    @abstractmethod
    def get_pos(self, color_pixel, depth_frame):
        pass

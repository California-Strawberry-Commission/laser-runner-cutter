import cv2

from furrow_perceiver.furrow_tracker import FurrowTracker
from furrow_perceiver.furrow_strip_tracker import FurrowStripTracker

MARKER_SIZE = 10
DISPLAY_LINES = True
BG_WIDTH = 140


class FurrowTrackerAnnotator:
    def __init__(self, tracker: FurrowTracker):
        self._tracker = tracker

    def annotate(self, img, draw_timings=False):
        # if draw_timings:
        #     cv2.rectangle(img, (0, 0), (BG_WIDTH, self.height), (127, 127, 0), -1)

        # Annotate strips
        for s in self._tracker.strips:
            self.annotate_strip(s, img, draw_timings)

        self.annotate_guidance(img)

    def annotate_guidance(self, img):
        tracker = self._tracker

        # Annotate pin
        if pin_x := tracker.get_reg_x(tracker.pin_y):
            cv2.drawMarker(
                img,
                (pin_x, tracker.pin_y),
                (0, 255, 0),
                markerType=cv2.MARKER_TRIANGLE_DOWN,
                markerSize=MARKER_SIZE * 2,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

            # Annotate guidance
            x = tracker.width // 2 + tracker.guidance_offset_x
            cv2.line(
                img,
                (x, tracker.height),
                (x, tracker.pin_y),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.arrowedLine(
                img,
                (x, tracker.pin_y),
                (pin_x, tracker.pin_y),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.5,
            )

        cv2.putText(
            img,
            f"{tracker.last_process_time * 1000:.1f}ms",
            (0, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if tracker.reg_intercept is not None:
            y0, y1 = 0, tracker.height
            x0 = tracker.get_reg_x(
                y0
            )  # (x0 - self.reg_intercept) / (self.reg_slope or 1)
            x1 = tracker.get_reg_x(y1)

            try:
                cv2.line(
                    img,
                    (int(x0), int(y0)),
                    (int(x1), int(y1)),
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                print("ERROR DRAWING LINE", x0, x1, y0, y1)
                raise Exception("sda")

    def annotate_strip(self, strip: FurrowStripTracker, img, display_timings=False):
        cv2.drawMarker(
            img,
            (strip.x_center, strip.y_center),
            (0, 0, 0),
            markerType=cv2.MARKER_STAR,
            markerSize=10,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

        cv2.drawMarker(
            img,
            (strip.x_center, strip.y_center),
            (0, 255, 0) if strip.is_valid else (0, 0, 255),
            markerType=cv2.MARKER_STAR,
            markerSize=10,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        # Draw furrow edge detections
        cv2.drawMarker(
            img,
            (strip.right_bound, strip.y_center),
            (0, 0, 255),
            markerType=cv2.MARKER_DIAMOND,
            markerSize=10,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        cv2.drawMarker(
            img,
            (strip.left_bound, strip.y_center),
            (0, 255, 0),
            markerType=cv2.MARKER_DIAMOND,
            markerSize=10,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        # Draw strip boundaries
        # cv2.line(
        #     img,
        #     (0, self.ymax),
        #     (self.width if DISPLAY_LINES else BG_WIDTH, self.ymax),
        #     color=(255, 255, 255),
        #     thickness=1,
        # )
        # cv2.line(
        #     img,
        #     (0, self.ymin),
        #     (self.width if DISPLAY_LINES else BG_WIDTH, self.ymin),
        #     color=(255, 255, 255),
        #     thickness=1,
        # )

        # fac, h, off = 0.4, 12, -2

        # # Display timings
        # if display_timings:
        #     cv2.putText(
        #         img,
        #         f"c {self.furrow_width:.1f}mm",
        #         (0, self.ymax - h + off),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         fac,
        #         (0, 0, 0),
        #         2,
        #     )

        #     cv2.putText(
        #         img,
        #         f"b {self.time_find_bounds * 1000:.2f}ms; c {self.time_convolve_strip * 1000:.2f}ms",
        #         (0, self.ymax + off),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         fac,
        #         (0, 0, 0),
        #         2,
        #     )

        #     # Display timings
        #     cv2.putText(
        #         img,
        #         f"c {self.furrow_width:.1f}mm",
        #         (0, self.ymax - h + off),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         fac,
        #         (255, 255, 255),
        #         1,
        #     )

        #     cv2.putText(
        #         img,
        #         f"b {self.time_find_bounds * 1000:.2f}ms; c {self.time_convolve_strip * 1000:.2f}ms",
        #         (0, self.ymax + off),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         fac,
        #         (255, 255, 255),
        #         1,
        #     )

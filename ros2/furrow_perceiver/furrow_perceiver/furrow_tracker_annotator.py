import cv2


class FurrowTrackerAnnotator:
    def annotate(self, img, draw_timings=False):
        if not self.width and not self.height:
            return
        
        if draw_timings:
            cv2.rectangle(img, (0, 0), (BG_WIDTH, self.height), (127, 127, 0), -1)
        
        # Annotate strips
        for s in self.strips:
            s.annotate(img, draw_timings)

        # Annotate pin
        if pin_x := self.get_reg_x(self.pin_y):
            cv2.drawMarker(
                img,
                ( pin_x, self.pin_y),
                (0, 255, 0),
                markerType=cv2.MARKER_TRIANGLE_DOWN,
                markerSize=MARKER_SIZE * 2,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        
            # Annotate guidance
            x = self.width // 2 + self.guidance_offset_x
            cv2.line(img, (x, self.height), (x, self.pin_y), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.arrowedLine(img, (x, self.pin_y), (pin_x, self.pin_y), (0, 255, 255), 2, cv2.LINE_AA, tipLength = 0.5)

        
        cv2.putText(
            img,
            f"{self.last_process_time * 1000:.1f}ms",
            (0, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if self.reg_intercept is not None:
            y0, y1 = 0, self.height
            x0 = self.get_reg_x(y0) # (x0 - self.reg_intercept) / (self.reg_slope or 1)
            x1 = self.get_reg_x(y1)

            try:
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1, cv2.LINE_AA)
            except Exception:
                print("ERROR DRAWING LINE", x0, x1, y0, y1)
                raise Exception("sda")
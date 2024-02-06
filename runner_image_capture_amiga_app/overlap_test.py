import cv2
import numpy as np
import pyrealsense2 as rs
import asyncio
import open3d as o3d


class RealSense:
    def __init__(self, frame_size=(1920, 1080), fps=30, camera_index=0):
        self.frame_size = frame_size
        self.camera_index = camera_index
        self.fps = fps

    def initialize(self):
        # Setup code based on https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        self.config = rs.config()

        # Connect to specific camera
        context = rs.context()
        devices = context.query_devices()
        if self.camera_index < 0 or self.camera_index >= len(devices):
            raise Exception("camera_index is out of bounds")

        serial_number = devices[self.camera_index].get_info(
            rs.camera_info.serial_number
        )
        self.config.enable_device(serial_number)

        # Configure stream
        self.config.enable_stream(
            rs.stream.color,
            self.frame_size[0],
            self.frame_size[1],
            rs.format.rgb8,
            self.fps,
        )

        # Start pipeline
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self.config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        if not frames:
            return None

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        return color_frame


camera = RealSense()
camera.initialize()


def rs_to_cv_frame(rs_frame):
    frame = np.asanyarray(rs_frame.get_data())
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

orb = cv2.ORB_create()


async def overlap_capture_task(stop_event: asyncio.Event):
    last_frame = None
    aggregated_shift = (0, 0)
    while not stop_event.is_set():
        await asyncio.sleep(1.0 / 10)
        current_frame = camera.get_frame()
        if current_frame:
            current_frame = rs_to_cv_frame(current_frame)
            if last_frame is not None:
                # Calculate overlap with last saved frame
                shift = calculate_overlap_3(last_frame, current_frame)
                """
                aggregated_shift = tuple(
                    map(lambda i, j: i + j, aggregated_shift, shift)
                )
                print(aggregated_shift)
                """
            last_frame = current_frame


def calculate_overlap_3(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Find features
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints1 = np.float32([kp.pt for kp in keypoints1]).reshape(-1, 1, 2)

    keypoints1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    # Run optical flow
    keypoints2, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, keypoints1, None, **lk_params
    )

    # Select good keypoints
    good_keypoints1 = []
    good_keypoints2 = []
    for i, (k1, k2) in enumerate(zip(keypoints1, keypoints2)):
        if status[i] == 1:
            good_keypoints1.append(k1)
            good_keypoints2.append(k2)
    good_keypoints1 = np.array(good_keypoints1)
    good_keypoints2 = np.array(good_keypoints2)

    # Draw tracks
    mask = np.zeros_like(frame1)  # Create a mask for visualization
    for i, (k1, k2) in enumerate(zip(good_keypoints1, good_keypoints2)):
        p1 = k1.ravel()
        p2 = k2.ravel()
        mask = cv2.line(
            mask, (int(p2[0]), int(p2[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2
        )
        frame2 = cv2.circle(frame2, (int(p2[0]), int(p2[1])), 5, (0, 0, 255), -1)

    # Overlay the optical flow tracks on the frame
    img = cv2.add(frame2, mask)
    cv2.imshow("Optical Flow with ORB Features", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_overlap_2(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the mean shift in x and y directions
    shift_x = int(flow[..., 0].mean())
    shift_y = int(flow[..., 1].mean())

    return (shift_x, shift_y)


def calculate_overlap(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Select the top N matches (you can adjust N based on your needs)
    N = min(len(matches), 50)
    selected_matches = matches[:N]

    # Extract corresponding keypoints
    src_points = np.float32(
        [keypoints1[m.queryIdx].pt for m in selected_matches]
    ).reshape(-1, 1, 2)
    dst_points = np.float32(
        [keypoints2[m.trainIdx].pt for m in selected_matches]
    ).reshape(-1, 1, 2)

    # Find the perspective transformation matrix
    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Get the dimensions of the input frames
    h, w = gray1.shape

    # Define the corners of the first frame
    corners1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
        -1, 1, 2
    )

    # Transform the corners using the perspective transformation matrix
    corners2 = cv2.perspectiveTransform(corners1, M)

    # Calculate the overlap percentage based on the transformed corners
    overlap_percentage = cv2.contourArea(corners2) / cv2.contourArea(corners1) * 100.0

    return overlap_percentage


async def main():
    stop_event = asyncio.Event()
    await overlap_capture_task(stop_event)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

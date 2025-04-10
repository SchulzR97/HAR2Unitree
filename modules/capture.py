import pyrealsense2 as rs
import cv2 as cv
import numpy as np

class RealSenseCamera():
    def __init__(
            self,
            max_w_h:int,
            serial_number:str = None
        ):
        self.__pipeline__ = rs.pipeline()
        self.__config__ = rs.config()
        if serial_number is not None:
            self.__config__.enable_device(serial_number)
        self.__pipeline_profile__ = self.__pipeline__.start(self.__config__)
        self.max_w_h = max_w_h

        self.align = rs.align(rs.stream.color)  

    def __del__(self):
        self.__pipeline__.stop()

    def __resize__(self, frame):
        f_h, f_w = frame.shape[0], frame.shape[1]
        if self.max_w_h is None:
            return frame
        scale = np.min([self.max_w_h / f_w, self.max_w_h / f_h])

        new_h, new_w = int(np.round(scale * f_h)), int(np.round(scale * f_w))
        frame = cv.resize(frame, (new_w, new_h))
        return frame

    def captureNext(self) -> tuple[np.array, np.array]:
        try:
            frames = self.__pipeline__.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if depth_frame and color_frame:
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                color_image = self.__resize__(color_image)
                depth_image = self.__resize__(depth_image)

                color_image = color_image / 255
                depth_image = depth_image / np.iinfo(np.uint16).max

                return color_image, depth_image
            return None, None
        finally:
            pass
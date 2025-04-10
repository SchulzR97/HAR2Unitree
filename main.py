from modules.capture import RealSenseCamera
from modules.action_recognition import HARProcessor
import cv2 as cv
import numpy as np

ACTION_LABELS = [
    'None',
    'Waving',
    'Pointing',
    'Clapping',
    'Follow',
    'Walking',
    'Stop',
    'Turn',
    'Jumping',
    'Come here',
    'Calm'
]

def action_detected(action, confidence):
    print(f"Action detected: A{action:0>3} - {ACTION_LABELS[action]} with confidence {confidence:.2f}")

if __name__ == '__main__':
    camera = RealSenseCamera(max_w_h=500)
    har_processor = HARProcessor(batch_size=1, input_size=(400, 400), use_depth_channel=False, k=3, on_action_detected=action_detected)

    while True:
        color_frame, depth_frame = camera.captureNext()
        if color_frame is None or depth_frame is None:
            continue

        har_processor.append(color_frame, depth_frame)


        #print(f'{har_processor.framerate:0.2f} FPS Action {har_processor.action:2>0} - {ACTION_LABELS[har_processor.action]} ({har_processor.confidence:.2f})')

        img_bgr = cv.cvtColor(np.asarray(color_frame.copy()*255, dtype=np.uint8), cv.COLOR_RGB2BGR) / 255

        cv.imshow('Color Image', img_bgr)
        cv.imshow('Depth Image', depth_frame)
        key = cv.waitKey(30)
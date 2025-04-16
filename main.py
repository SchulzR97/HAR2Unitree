from modules.capture import RealSenseCamera
from modules.action_recognition import HARProcessor
from modules.visualization import HAR2UnitreeUI
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
    if action == 0:
        return
    print(f"Action detected: A{action:0>3} - {ACTION_LABELS[action]} with confidence {confidence:.2f}")

if __name__ == '__main__':
    camera = RealSenseCamera(max_w_h=1500)
    har_processor = HARProcessor(
        batch_size=10,#10
        input_size=(400, 400),
        use_depth_channel=False,
        min_confidence=0.4,
        target_framerate=28,
        pred_i=14,#15
        k=1,
        on_action_detected=action_detected
    )
    ui = HAR2UnitreeUI(ACTION_LABELS)

    while True:
        color_frame, depth_frame = camera.captureNext()
        if color_frame is None or depth_frame is None:
            continue

        har_processor.append(color_frame, depth_frame)

        ui.detected = har_processor.detected
        ui.action = har_processor.action
        ui.confidence = har_processor.confidence
        ui.framerate = har_processor.framerate
        ui.scores = har_processor.scores.numpy()
        ui.buffer_size = len(har_processor.buffer)

        ui.show(color_frame)
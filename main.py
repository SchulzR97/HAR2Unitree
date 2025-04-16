from modules.capture import RealSenseCamera
from modules.action_recognition import HARProcessor
from modules.visualization import HAR_UI, Robot_UI
from modules.unitree import UnitreeGo1
import cv2 as cv
import numpy as np
from enum import Enum

class ActionLabel(Enum):
    NONE = 0
    Waving = 1
    Pointing = 2
    Clapping = 3
    Follow = 4
    Walking = 5
    Stop = 6
    Turn = 7
    Jumping = 8
    Come_here = 9
    Calm = 10

class State():
    DAMPING = 0
    IDLE = 1
    TRACKING = 2

if __name__ == '__main__':
    action_labels = [action_label.name for action_label in ActionLabel]

    camera = RealSenseCamera(max_w_h=1500)
    har_processor = HARProcessor(
        batch_size=10,#10
        input_size=(400, 400),
        use_depth_channel=False,
        min_confidence=0.4,
        target_framerate=28,
        pred_i=14,#15
        k=1,
        #on_action_detected=action_detected
    )
    go1 = UnitreeGo1()

    go1.process_async(go1.process_stand_up, ())
    state = State.IDLE

    har_ui = HAR_UI(action_labels)
    # robot_ui = Robot_UI(
    #     'Robot UI',
    #     func_move_foward=lambda: go1.process_async(go1.move_forward, ()),
    #     func_turn=lambda angle: go1.process_async(go1.turn, (angle,)),
    #     func_joy=lambda: go1.process_async(go1.process_joy, ()),
    #     func_stand_up=lambda: go1.process_async(go1.process_stand_up, ()),
    #     func_damping=lambda: go1.process_async(go1.process_damp_all_motors, ()),
    # )

    while True:
        color_frame, depth_frame = camera.captureNext()
        if color_frame is None or depth_frame is None:
            continue

        har_processor.append(color_frame, depth_frame)

        action_label = ActionLabel(har_processor.action)
        if har_processor.detected and go1.progress is None:
            if state == State.DAMPING:
                if action_label == ActionLabel.Follow:
                    go1.process_async(go1.process_stand_up, ())
            elif state == State.IDLE:
                if action_label == ActionLabel.Clapping:
                    go1.process_async(go1.process_joy, ())
                elif action_label == ActionLabel.Pointing:
                    go1.process_async(go1.process_damp_all_motors, ())
                    state = State.DAMPING
                elif action_label == ActionLabel.Waving:
                    go1.process_async(go1.dance1, ())
                elif action_label == ActionLabel.Turn:
                    go1.process_async(go1.turn, (90,))

        if go1.progress is None and state != State.DAMPING:
            state = State.IDLE


        har_ui.detected = har_processor.detected
        har_ui.action = har_processor.action
        har_ui.confidence = har_processor.confidence
        har_ui.framerate = har_processor.framerate
        har_ui.scores = har_processor.scores.numpy()
        har_ui.buffer_size = len(har_processor.buffer)

        har_ui.show(color_frame)
        #robot_ui.__render__()
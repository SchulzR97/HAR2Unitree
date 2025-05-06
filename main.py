from modules.capture import RealSenseCamera
from modules.action_recognition import HARProcessor
from modules.visualization import HAR_UI
from modules.unitree import UnitreeGo1
from modules.person_tracking import PersonTracker
from modules.framework import HAR2UnitreeProcessor

if __name__ == '__main__':
    camera = RealSenseCamera(max_w_h=1500)

    har_processor = HARProcessor(
        batch_size=4,#10
        input_size=(400, 400),
        use_depth_channel=False,
        min_confidence=0.6,
        target_framerate=28,#28
        pred_i=10,#15
        k=1,
        #on_action_detected=action_detected
    )

    go1 = UnitreeGo1()
    go1.process_async(go1.stand_up, ())

    person_tracker = PersonTracker(
        alpha=0.86,#0.88
        beta=0.96,#0.96
        tau=0.85,#0.867
        N=20#20
    )

    har2unitree = HAR2UnitreeProcessor(
        camera=camera,
        go1=go1,
        har_processor=har_processor,
        person_tracker=person_tracker
    )
    har_ui = HAR_UI(har2unitree)

    har2unitree.start()

    while True:
        har_ui.show()
        #robot_ui.__render__()
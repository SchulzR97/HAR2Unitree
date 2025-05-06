from modules.action_recognition import HARProcessor
from modules.capture import RealSenseCamera
from modules.unitree import UnitreeGo1
from modules.person_tracking import PersonTracker
from modules.stats import ActionLabel, RobotState
from threading import Thread

class HAR2UnitreeProcessor():
    NEXT_ACTIONS = {
        RobotState.DAMPING: [ActionLabel.Follow],
        RobotState.IDLE: [
            ActionLabel.Waving,
            ActionLabel.Pointing,
            ActionLabel.Clapping,
            ActionLabel.Follow,
            ActionLabel.Turn,
            ActionLabel.Jumping,
            ActionLabel.Come_here,
            ActionLabel.Calm
        ],
        RobotState.TRACKING: [
            ActionLabel.Stop
        ],
        RobotState.JOY: [],
        RobotState.DANCING: [],
        RobotState.STAND_UP: [],
        RobotState.TURNING: [],
        RobotState.MOVE_FORWARD: [],
        RobotState.MOVE_BACKWARD: [],
        RobotState.JUMPING: []
    }

    def __init__(
            self,
            camera:RealSenseCamera,
            go1:UnitreeGo1,
            har_processor:HARProcessor,
            person_tracker:PersonTracker
    ):
        self.camera = camera
        self.go1 = go1
        self.har_processor = har_processor
        self.person_tracker = person_tracker
        self.tracking_results = []

        # states
        self.tracking_results = []
        self.running = False
        self.disposing = False
        
    def start(self):
        self.thread = Thread(target=self.cycle)
        self.thread.start()
        self.running = True

    def dispose(self):
        self.disposing = True
    
    def cycle(self):
        self.go1.process_async(self.go1.stand_up, ())

        while not self.disposing:
            color_frame, depth_frame = self.camera.captureNext()
            if color_frame is None or depth_frame is None:
                continue

            self.color_frame = color_frame
            self.har_processor.append(color_frame, depth_frame)

            action_label = ActionLabel(self.har_processor.action)
            if self.har_processor.detected and self.go1.progress is None:
                if self.go1.state == RobotState.DAMPING:
                    if action_label == ActionLabel.Follow:
                        self.go1.process_async(self.go1.stand_up, ())
                elif self.go1.state == RobotState.IDLE:
                    if action_label == ActionLabel.Clapping:
                        self.go1.process_async(self.go1.joy, ())
                    elif action_label == ActionLabel.Calm:
                        self.go1.process_async(self.go1.damp_all_motors, ())
                    elif action_label == ActionLabel.Waving:
                        self.go1.process_async(self.go1.dance1, ())
                    elif action_label == ActionLabel.Turn:
                        self.go1.process_async(self.go1.turn360, ())
                    elif action_label == ActionLabel.Follow:
                        self.go1.state = RobotState.TRACKING
                        self.person_tracker.initialize_target_subject(color_frame)
                    elif action_label == ActionLabel.Come_here:
                        self.go1.process_async(self.go1.move_forward, (0.3,))
                    elif action_label == ActionLabel.Pointing:
                        self.go1.process_async(self.go1.move_backward, (0.2,))
                    elif action_label == ActionLabel.Jumping:
                        self.go1.process_async(self.go1.jump, ())
                elif self.go1.state == RobotState.TRACKING:
                    results = self.person_tracker(color_frame)
                    self.tracking_results = results

                    for result in results:
                        detected = result['detected']
                        if detected:
                            x1, y1, x2, y2 = result['bbox']
                            cx = int(x1 + (x2 - x1) / 2)
                            img_center_x = int(color_frame.shape[1] / 2)
                            diff = img_center_x - cx
                            diff_rel = diff / (color_frame.shape[1] / 2)
                            #print(f'diff: {diff_rel}')
                            if abs(diff_rel) > 0.2:
                                self.go1.turn(diff_rel * 0.0001)
                                self.go1.state = RobotState.TRACKING
                            pass
                    pass
            if self.go1.state == RobotState.TRACKING and action_label == ActionLabel.Stop:
                self.go1.state = RobotState.IDLE
                self.tracking_results = []

        self.running = False
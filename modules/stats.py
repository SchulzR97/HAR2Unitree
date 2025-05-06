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

class RobotState(Enum):
    DAMPING = 0
    IDLE = 1
    TRACKING = 2
    JOY = 3
    DANCING = 4
    STAND_UP = 5
    TURNING = 6
    MOVE_FORWARD = 7
    MOVE_BACKWARD = 8
    JUMPING = 9
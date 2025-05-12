import sys
import time
import math
from threading import Thread
from modules.stats import RobotState

sys.path.append('/home/messe/Documents/GIT/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

class Mode():
    """
    idle = 0<br>
    standing_in_force_control = 1<br>
    walking_following_target_velocity = 2<br>
    walking_following_target_position = 3   # reserve for future release<br>
    walking_following_a_given_path = 4      # reserve for future release<br>
    stand_down_in_position_control = 5<br>
    stand_up_in_position_control = 6<br>
    damping_mode_all_motors = 7<br>
    recovery_mode = 8<br>
    backflip = 9<br>
    jumpYaw = 10<br>
    straightHand = 11<br>
    dance1 = 12<br>
    dance2 = 13
    """
    idle = 0
    standing_in_force_control = 1
    walking_following_target_velocity = 2
    walking_following_target_position = 3   # reserve for future release
    walking_following_a_given_path = 4      # reserve for future release
    stand_down_in_position_control = 5
    stand_up_in_position_control = 6
    damping_mode_all_motors = 7
    recovery_mode = 8
    backflip = 9
    jumpYaw = 10
    straightHand = 11
    dance1 = 12
    dance2 = 13

class GaitType():
    idle = 0
    trot_walking = 1
    trot_running = 2
    stairs_climbing = 3
    trot_obstacle = 4

class SpeedLevel():
    '''
    Only used for mode 3
    '''
    default_low_speed = 0
    default_medium_speed = 1
    default_high_speed = 2

class LowCmd(sdk.LowCmd):
    def __init__(
            self
    ):
        super().__init__()

class HighCmd(sdk.HighCmd):
    def __init__(
            self,
            mode:Mode = Mode.idle,
            gait_type:GaitType = GaitType.idle,
            speed_level:SpeedLevel = SpeedLevel.default_low_speed,
            footRaiseHeight:float = 0.08,
            bodyHeight:float = 0.28,
            position:tuple[float, float] = [0, 0],
            euler:tuple[float, float, float] = [0, 0, 0],
            velocity:tuple[float, float] = [0, 0],
            yawSpeed:float = 0.,
            reserve:int = 0
        ):
        """
        unitree.HighCmd
        ---------------
        High level command for Unitree Go1 interaction. See documentation: https://unitree-docs.readthedocs.io/en/latest/get_started/Go1_Edu.html

        Parameters
        ----------
        mode : unitree.Mode (Enum)
            - idle = 0
            - standing_in_force_control = 1
            - walking_following_target_velocity = 2
            - walking_following_target_position = 3 -> reserve for future release
            - walking_following_a_given_path = 4 -> reserve for future release
            - stand_down_in_position_control = 5
            - stand_up_in_position_control = 6
            - damping_mode_all_motors = 7
            - recovery_mode = 8
            - backflip = 9
            - jumpYaw = 10
            - straightHand = 11
            - dance1 = 12
            - dance2 = 13
        gaitType : unitree.GaitType (Enum)
            - idle = 0
            - trot_walking = 1
            - trot_running = 2
            - stairs_climbing = 3
            - trot_obstacle = 4
        speedLevel : unitree.SpeedLevel (Enum)
            SpeedLevel setting is now only used for mode 3.
            - default_low_speed = 0
            - default_medium_speed = 1
            - default_high_speed = 2
        footRaiseHeight : float, default 0.08
            Swing foot height adjustment from default swing height.delta valuedefault: 0.08m
        bodyHeight : float, default 0.28
            Body height adjustment from default body height.delta valuedefault: 0.28m
        position : tuple[float, float], default [0.0, 0.0]
            Desired x and y position in the inertial frame, which is established at the beginning instance of the sport mode. Position setting is used in mode 3 as target position.
        euler : tuple[float, float, float], default [0.0, 0.0, 0.0]
            Desired yaw-pitch-roll Euler angle, with euler[0] = Roll,euler[1] = Pitch,euler[2] = Yaw.RPY setting can be used in mode 1 as target orientation.Yaw setting can be used in mode 3 as target yaw angle.
        velocity : tuple[float, float], default [0.0, 0.0]
            Desired robot forward speed and side speed in the body frame. Velocity setting is used in mode 2 as target linear velocity.
        yawSpeed : float, default 0.0
            Desired rotational yaw speed. YawSpeed setting is used in mode 2 as target rotational speed.
        """
        super().__init__()
        self.mode:Mode = mode
        self.gaitType:GaitType = gait_type
        self.speedLevel:int = speed_level
        self.footRaiseHeight:int = footRaiseHeight
        self.bodyHeight:float = bodyHeight
        self.position:tuple[float, float] = position
        self.euler:tuple[float, float] = euler
        self.velocity:tuple[float, float] = velocity
        self.yawSpeed = yawSpeed
        self.reserve = reserve

class LevelFlag():
    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

class HighState(sdk.HighState):
    def __init__(self):
        super().__init__()

class UnitreeGo1():
    def __init__(
            self,
            client_port:int = 8080,
            server_ip:str = "192.168.123.161",
            server_port:int = 8082
    ):
        """
        Unitree Go1 interaction
        
        Parameters
        ----------
        client_port : int, default
            Port of the client
        server_ip : str, default "192.168.123.10"
            IP address of the robot
        server_port : int, default 8007
            Port of the server
        """
        self.client_port = client_port
        self.server_ip = server_ip
        self.server_port = server_port
        self.udp_high_level = sdk.UDP(LevelFlag.HIGHLEVEL, client_port, server_ip, server_port)
        self.udp_low_level = sdk.UDP(LevelFlag.LOWLEVEL, 8007, server_ip, server_port)
        self.progress = None
        self.state = RobotState.DAMPING

        self.__command_level__ = LevelFlag.HIGHLEVEL
        self.__next_turn_direction__ = 1

    def joy(self):
        if self.progress is not None:
            return
        self.progress = 0.
        self.state = RobotState.JOY
        num = 5
        for i in range(num):
            cmd = HighCmd(mode = Mode.standing_in_force_control, euler=[-0., -0.3, 0])
            self.send_HighCmd(cmd)
            time.sleep(0.5)

            self.progress = (2*i+1) / (2 * num)

            cmd = HighCmd(mode = Mode.standing_in_force_control, euler=[0., 0.3, 0])
            self.send_HighCmd(cmd)
            time.sleep(0.5)
            self.progress = (2*i+2) / (2 * num)
        self.progress = None
        self.state = RobotState.IDLE
    
    def stand_up(self):
        """mode7 -> mode5 -> mode6
        """
        if self.progress is not None:
            return
        self.progress = 0.
        self.state = RobotState.STAND_UP
        cmd = HighCmd(mode = Mode.stand_up_in_position_control)
        self.send_HighCmd(cmd)
        for i in range(20):
            self.progress = (i+1) / 20
            time.sleep(0.1)
        self.progress = None
        self.state = RobotState.IDLE

    def jump(self):
        if self.progress is not None:
            return
        self.progress = 0.
        self.state = RobotState.JUMPING
        num = 2
        for i in range(num):
            cmd = HighCmd(mode = Mode.stand_down_in_position_control, bodyHeight=0.28)
            self.send_HighCmd(cmd)
            time.sleep(0.5)

            cmd = HighCmd(mode = Mode.stand_up_in_position_control, bodyHeight=0.4)
            self.send_HighCmd(cmd)
            time.sleep(0.5)

            self.progress = (i+1) / (num)

            # cmd = HighCmd(mode = Mode.standing_in_force_control, euler=[0., 0.3, 0])
            # self.send_HighCmd(cmd)
            # time.sleep(0.5)
            # self.progress = (2*i+2) / (2 * num)
        self.progress = None
        self.state = RobotState.IDLE

    def damp_all_motors(self):
        if self.progress is not None:
            return
        self.progress = 0.
        self.state = RobotState.DAMPING
        cmd = HighCmd(mode = Mode.stand_down_in_position_control)
        self.send_HighCmd(cmd)
        for i in range(4):
            time.sleep(0.2)
            self.progress = (i+1) / 6
        cmd = HighCmd(mode = Mode.damping_mode_all_motors)
        self.send_HighCmd(cmd)
        for i in range(5, 6):
            time.sleep(0.5)
            self.progress = ((i+1)) / 6
        self.progress = None
        self.state = RobotState.DAMPING

    def dance1(self):
        if self.progress is not None:
            return
        self.progress = 0.
        self.state = RobotState.DANCING

        cmd = HighCmd(mode = Mode.walking_following_target_velocity)
        self.send_HighCmd(cmd)
        time.sleep(0.5)

        cmd = HighCmd(mode = Mode.dance1)
        
        self.send_HighCmd(cmd)
        seconds = 17
        for i in range(seconds * 2):
            time.sleep(0.5)
            self.progress = (i + 1) / (seconds * 2)
        self.progress = None
        self.state = RobotState.IDLE

    def dance2(self):
        if self.progress is not None:
            return
        self.progress = 0.
        self.state = RobotState.DANCING
        cmd = HighCmd(mode = Mode.dance1)
        
        self.send_HighCmd(cmd)
        seconds = 37
        for i in range(seconds * 2):
            time.sleep(0.5)
            self.progress = (i + 1) / (seconds * 2)
        self.progress = None
        self.state = RobotState.IDLE

    def turn360(self):
        self.turn(self.__next_turn_direction__ * 360 * 1.05, 1.5)
        self.__next_turn_direction__ = -self.__next_turn_direction__

    def turn(self, angle_deg, angular_velocity = 0.25):
        if self.progress is not None:
            return
        self.progress = 0.
        self.state = RobotState.TURNING
        angle_rad = math.radians(angle_deg)
        time_required = abs(angle_rad / angular_velocity) #* 1.1
        
        cmd = HighCmd(
            mode = Mode.walking_following_target_velocity,
            gait_type = GaitType.idle,
            speed_level = SpeedLevel.default_low_speed,
            yawSpeed = angular_velocity if angle_rad > 0 else -angular_velocity,
            #velocity=[0.05, 0.03]
            velocity=[0.0, 0.0]
        )

        start_time = time.time()
        while time.time() - start_time < time_required:
            self.send_HighCmd(cmd)
            time.sleep(0.01)
            self.progress = (time.time() - start_time) / time_required
        self.progress = None
        self.state = RobotState.IDLE

    def move_forward(self, distance = 0.2):
        if self.progress is not None:
            return
        self.progress = 0
        self.state = RobotState.MOVE_FORWARD

        n = int(distance / 0.2)
        n = min([1, n])

        cmd = HighCmd(
            mode = Mode.walking_following_target_velocity,
            gait_type = GaitType.trot_walking,
            speed_level = SpeedLevel.default_low_speed,
            velocity = [0.2, 0.02],
            euler=[0., 0., 0.]
        )
        for i in range(n):
            self.send_HighCmd(cmd)
            for j in range(10):
                self.progress = ((i+1) * 10  + j / 10) / (n * 10)
                time.sleep(0.1)
        self.progress = None
        self.state = RobotState.IDLE

    def move_backward(self, distance = 0.2):
        if self.progress is not None:
            return
        self.progress = 0
        self.state = RobotState.MOVE_BACKWARD

        n = int(distance / 0.2)
        n = min([1, n])

        cmd = HighCmd(
            mode = Mode.walking_following_target_velocity,
            gait_type = GaitType.trot_walking,
            speed_level = SpeedLevel.default_low_speed,
            velocity = [-0.2, 0.],
            euler=[0., 0., 0.]
        )
        for i in range(n):
            self.send_HighCmd(cmd)
            for j in range(10):
                self.progress = ((i+1) * 10  + j / 10) / (n * 10)
                time.sleep(0.1)
        self.progress = None
        self.state = RobotState.IDLE

    def process_async(self, target, args):
        Thread(target=target, args=args).start()

    def send_HighCmd(self, cmd:HighCmd):
        if self.__command_level__ != LevelFlag.HIGHLEVEL:
            raise ValueError("Command level is not HIGHLEVEL")
        
        self.udp_high_level.InitCmdData(cmd)

        state_base = sdk.HighState()
        recv = self.udp_high_level.Recv()
        get_recv = self.udp_high_level.GetRecv(state_base)

        self.udp_high_level.SetSend(cmd)
        send = self.udp_high_level.Send()

        if send != 129:
            pass

    def send_LowCmd(self, cmd:LowCmd):
        if self.__command_level__ != LevelFlag.LOWLEVEL:
            raise ValueError("Command level is not LOWLEVEL")
        
        self.udp_low_level.InitCmdData(cmd)

        state_base = sdk.LowState()
        recv = self.udp_low_level.Recv()
        get_recv = self.udp_low_level.GetRecv(state_base)

        self.udp_low_level.SetSend(cmd)
        send = self.udp_low_level.Send()

        if send != 129:
            pass

    def switch_command_level(self, level:LevelFlag):
        cmd = HighCmd()
        cmd.mode = level

        self.progress = 0.

        for i in range(100):
            self.udp_high_level.InitCmdData(cmd)

            state_base = sdk.HighState()
            recv = self.udp_high_level.Recv()
            get_recv = self.udp_high_level.GetRecv(state_base)

            self.udp_high_level.SetSend(cmd)
            send = self.udp_high_level.Send()

            if send != 129:
                pass

            time.sleep(0.01)
            self.progress = (i+1) / 100

        self.progress = None
        self.__command_level__ = level
import numpy as np
import torch
import cv2 as cv
from threading import Thread
from datetime import datetime
from rsp.ml.model import load_model
import time
from collections import deque

class TriggerHL():
    def __init__(
            self,
            delay:int
    ):
        self.delay = delay
        self.cnt = 0
        self.active_action = 0
        self.last_action = 0

    def __call__(self, action:int):
        if action == self.last_action:
            self.cnt += 1

        self.last_action = action

        if self.cnt >= self.delay:
            self.active_action = action
            self.cnt = 0
        return self.active_action

class HARProcessor():
    def __init__(
            self,
            batch_size:int = 1,
            sequence_length:int = 30,
            input_size:tuple = (400, 400),
            use_depth_channel:bool = False,
            pred_i:int = 30,
            target_framerate:int = 30,
            k:int = 1,
            on_action_detected:callable = None,
            min_confidence:float = 0.5
    ):
        self.use_depth_channel = use_depth_channel
        self.input_size = input_size
        self.pred_i = pred_i
        self.target_framerate = target_framerate
        self.k = k
        self.on_action_detected = on_action_detected
        self.min_confidence = min_confidence

        self.trigHL = TriggerHL(2)

        self.X = torch.zeros((batch_size, sequence_length, 4 if use_depth_channel else 3, input_size[0], input_size[1]), dtype=torch.float32)

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model = load_model(
            user_id='SchulzR97',
            model_id='MSCONV3Ds',
            weights_id='TUC-HRI-LAB'
        )
        #self.model = torch.jit.load('model/MSCONV3Ds/TUC-HRI-LAB.pth')
        self.model.eval()
        self.model.to(self.device)

        self.buffer = []
        self.i = 0
        self.detected = False
        self.action = 0
        self.confidence = 0.
        self.framerate = target_framerate
        self.last_frame_time = time.time()
        self.scores = torch.zeros_like(self.model.readout.bias).cpu()

        self.__prediction_data__ = {
            'last_action': 0,
            'conf': 0.
        }

        self.__thread__ = Thread(target=self.__thread_cycle__)
        self.__thread__.start()
        pass

    def predict(self):
        with torch.no_grad():
            self.X = self.X.to(self.device)
            pred = self.model(self.X).detach().cpu()
            decay = torch.linspace(0.2, 1., steps=self.X.shape[0])

            pred_decay = pred * decay.unsqueeze(1)

            action_rebalance = torch.tensor([
                1.3, # None
                0.9, # Waving
                1.5, # Pointing
                1.1, # Clapping
                0.8, # Follow
                1.0, # Walking
                2.7, # Stop
                1.3, # Turn
                1.0, # Jumping
                1.1, # Come here
                2.9  # Calm
            ], dtype=torch.float32)
            pred_decay = pred_decay * action_rebalance

            #self.scores = torch.sum(pred_decay, dim=0) / torch.sum(pred_decay)
            self.scores = torch.nn.functional.softmax(pred_decay.sum(dim=0), dim=0)
            
            confidence = torch.max(self.scores).item()
            self.detected = confidence >= self.min_confidence

            if self.detected:
                action = torch.argmax(self.scores).item()
                self.action = self.trigHL(action)
                self.confidence = confidence
                self.on_action_detected(self.action, self.confidence)
            else:
                self.action = 0
                self.confidence = 0.

            # self.action = torch.argmax(scores).item()
            # self.confidence = torch.max(scores).item()
            # self.detected = self.confidence >= self.min_confidence
            # if self.detected:
            #     self.on_action_detected(self.action, self.confidence)
            # else:
            #     self.action = 0
            #     self.confidence = 0.


            # action = torch.argmax(scores, dim=1)
            # confidence, _ = torch.max(scores, dim=1)

            # actions, cnt = torch.unique(action, return_counts=True)
            # scores = cnt / torch.sum(cnt)

            # i = torch.argmax(scores)
            # self.action = actions[i].item()
            # self.confidence = scores[i].item()
            # self.detected = self.confidence >= self.min_confidence

    def __thread_cycle__(self):
        while True:
            if len(self.buffer) > 0:
                frame = self.buffer.pop(0)

                # move batch
                if self.i%self.k == 0:
                    self.X[:-1, :-1] = self.X[1:, 1:].clone()
                
                # move frames
                self.X[-1, :-1] = self.X[-1, 1:].clone()

                # append frame
                self.X[:-1] = self.X[1:].clone()

                frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
                self.X[-1, -1] = frame_tensor

                if self.i%self.pred_i == 0:
                    Thread(target=self.predict).start()

                self.i += 1
            time.sleep(0.001)

    def append(
            self,
            color_frame:np.array,
            depth_frame:np.array
    ):
        color_frame = cv.resize(color_frame, self.input_size)
        if self.use_depth_channel:
            depth_frame = cv.resize(depth_frame, self.input_size)

        #color_frame = cv.cvtColor(np.array(color_frame * 255, dtype=np.uint8), cv.COLOR_RGB2BGR) / 255
        
        frame = color_frame
        if self.use_depth_channel:
            frame = np.concatenate((frame, depth_frame), axis=-1)

        while time.time() - self.last_frame_time < 1 / self.target_framerate:
            time.sleep(0.001)

        new_framerate = 1 / (time.time() - self.last_frame_time)
        self.framerate += 5e-1 * (new_framerate - self.framerate)

        self.buffer.append(frame)
        self.last_frame_time = time.time()

        assert len(self.buffer) <= 20, f'Buffer size exceeded: {len(self.buffer)} > 20'
import numpy as np
import torch
import cv2 as cv
from threading import Thread
from datetime import datetime
from rsp.ml.model import load_model
import time

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

        self.X = torch.zeros((batch_size, sequence_length, 4 if use_depth_channel else 3, input_size[0], input_size[1]), dtype=torch.float32)

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model = load_model(
            user_id='SchulzR97',
            model_id='MSCONV3D',
            weights_id='TUC-HRI'
        )
        self.model.to(self.device)

        self.buffer = []
        self.i = 0
        self.action = 0
        self.confidence = 0.
        self.framerate = target_framerate
        self.last_frame_time = time.time()

        self.__prediction_data__ = {
            'last_action': 0,
            'k': 0
        }

        self.__thread__ = Thread(target=self.__thread_cycle__)
        self.__thread__.start()
        pass

    def predict(self):
        with torch.no_grad():
            self.X = self.X.to(self.device)
            pred = self.model(self.X)
            action = torch.argmax(pred, dim=1).item()
            confidence = torch.max(pred, dim=1).values.item()
            
            if confidence > self.min_confidence:
                if action == self.__prediction_data__['last_action']:
                    self.__prediction_data__['k'] += 1
                else:
                    self.__prediction_data__['k'] = 1
                self.__prediction_data__['last_action'] = action
            else:
                self.__prediction_data__['k'] = 0
                self.__prediction_data__['last_action'] = 0
            
            if self.__prediction_data__['k'] >= self.k:
                self.action = action
                self.confidence = confidence
                if self.on_action_detected is not None:
                    self.on_action_detected(self.action, self.confidence)
            else:
                self.action = 0
                self.confidence = 0.

    def __thread_cycle__(self):
        while True:
            if len(self.buffer) > 0:
                frame = self.buffer.pop(0)
                
                self.X[-1, :-1] = self.X[-1, 1:].clone()
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
        
        frame = color_frame
        if self.use_depth_channel:
            frame = np.concatenate((frame, depth_frame), axis=-1)

        while time.time() - self.last_frame_time < 1 / self.target_framerate:
            time.sleep(0.001)

        new_framerate = 1 / (time.time() - self.last_frame_time)
        self.framerate += 5e-1 * (new_framerate - self.framerate)

        self.buffer.append(frame)
        self.last_frame_time = time.time()
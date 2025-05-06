import torchreid
import torch
import numpy as np
from torchvision import transforms
from torch.nn.functional import cosine_similarity
from ultralytics import YOLO
from collections import deque
from PIL import Image

class PersonTracker():
    def __init__(
        self,
        alpha:float = 0.7,
        beta:float = 0.8,
        tau:float = 0.7,
        N:int = 10
    ):
        self.crop_ratio = (128, 256)
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.N = N

        # feature extraction
        self.E_phi = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=751,
            pretrained=True
        )
        self.E_phi.classifier = torch.nn.Flatten()
        self.E_phi.eval()

        # bounding box prediction
        self.bb = YOLO('yolov8n.pt')

        # preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # device
        if torch.cuda.is_available():
            self.E_phi.to('cuda')
            self.bb.to('cuda')
        elif torch.backends.mps.is_available():
            self.E_phi.to('mps')
            self.bb.to('mps')

    def __call__(self, frame:np.array):
        frame = np.array(frame.copy() * 255, dtype=np.uint8)
        results = []

        B = self.__detect_persons__(frame)

        s_pos_max = -99999

        features = []

        for i in range(len(B)):
            b_i = B[i]

            # crop person
            X_i = self.__crop__(frame, b_i)

            # extract features
            phi_i = self.__extract_features__(X_i)
            features.append(phi_i)

            # calculate similarity
            s_i_pos = self.__calc_similarity__(phi_i, self.Phi_pos)
            s_i_neg = self.__calc_similarity__(phi_i, self.Phi_neg)
            s_i_pos_max = np.max(s_i_pos)
            s_i_neg_max = np.max(s_i_neg)

            if s_i_pos_max > s_pos_max:
                s_pos_max = s_i_pos_max
            
            results.append({
                's_pos_max': s_i_pos_max,
                's_neg_max': s_i_neg_max,
                'bbox': b_i
            })

        for result, phi_i in zip(results, features):
            s_i_pos_max = result['s_pos_max']
            s_i_neg_max = result['s_neg_max']
            diff = s_i_pos_max - s_i_neg_max

            # evaluate
            diff = s_i_pos_max - s_i_neg_max
            if s_i_pos_max >= self.tau:         # positive feature set similarity is higher than detection threshold tau
                if len(self.Phi_neg) == 0:      # ignore diff, if negative features are empty
                    result['detected'] = True
                elif diff > 0.01 and\
                    s_i_pos_max == s_pos_max:               # diff is positive (positive feature set similarity is higher
                                                # than negative feature set similarity)
                    result['detected'] = True
                else:
                    result['detected'] = False
            else:
                result['detected'] = False

            # update features sets
            if result['detected'] and s_i_pos_max >= self.alpha and s_i_pos_max <= self.beta and s_i_pos_max == s_pos_max:
                self.Phi_pos.append(phi_i)
            elif not result['detected'] and (len(self.Phi_neg) == 0 or diff < -0.05):#or s_i_neg_max >= self.alpha and s_i_neg_max <= self.beta
                self.Phi_neg.append(phi_i)
        
        return results

    def reset(self):
        self.Phi_pos = deque(maxlen=self.N)
        self.Phi_neg = deque(maxlen=self.N)

    def initialize_target_subject(self, img:np.array):
        img = np.array(img.copy() * 255, dtype=np.uint8)

        self.reset()
        bboxes = self.__detect_persons__(img)

        center = (img.shape[1] // 2, img.shape[0] // 2)
        # find most central bounding box
        min_dist, best_bbox = 99999, None
        for bbox in bboxes:
            box_center = self.__box_center__(bbox)
            dist = self.__distance__(box_center, center)
            if dist < min_dist:
                min_dist = dist
                best_bbox = bbox

        if best_bbox is None:
            raise ValueError("No person detected in the image.")

        initial_cropped = self.__crop__(img, best_bbox)
        features = self.__extract_features__(initial_cropped)
        self.Phi_pos.append(features)

    def __detect_persons__(self, img):
        results = self.bb(img, conf = 0.5, verbose=False)

        boxes = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])                   # class index person
                if cls != 0:
                    continue

                confidence = box.conf[0]                # confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates

                boxes.append((x1, y1, x2, y2))

        return boxes

    def __extract_features__(self, img):
        self.E_phi.eval()

        img = self.preprocess(Image.fromarray(img).convert("RGB")).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.to('cuda')
        elif torch.backends.mps.is_available():
            img = img.to('mps')

        with torch.no_grad():
            features = self.E_phi(img)

        return features

    def __distance__(self, p1, p2):
        dist = np.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)
        return dist

    def __calc_similarity__(self, phi_i, Phi_comp):
        similarities = []
        for phi_comp in Phi_comp:
            similarity = cosine_similarity(phi_i, phi_comp).item()
            similarities.append(similarity)

        if len(similarities) == 0:
            similarities = [np.nan]

        return similarities

    def __crop__(self, img, bbox):
        crop_ratio = self.crop_ratio[1] / self.crop_ratio[0]
        bbox_ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])

        # expand bounding box in h
        if bbox_ratio < crop_ratio:
            w = bbox[2] - bbox[0]
            h = int(np.round(crop_ratio * w))
        # expand bounding box in w
        elif bbox_ratio > crop_ratio:
            h = bbox[3] - bbox[1]
            w = int(np.round(h / crop_ratio))
        # bounding box has target ratio -> take it as it is
        else:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

        center = self.__box_center__(bbox)

        if center[0] - w // 2 < 0:
            w = center[0] * 2
            h = int(np.round(crop_ratio * w))
        if center[0] + w // 2 >= img.shape[1]:
            w = (img.shape[1] - center[0]) * 2
            h = int(np.round(crop_ratio * w))
        if center[1] - h // 2 < 0:
            h = center[1] * 2
            w = int(np.round(h / crop_ratio))
        if center[1] + h // 2 >= img.shape[0]:
            h = (img.shape[0] - center[1]) * 2
            w = int(np.round(h / crop_ratio))

        sx = center[0] - w // 2
        ex = sx + w
        sy = center[1] - h // 2
        ey = sy + h

        img_cropped = img[sy:ey, sx:ex]

        return img_cropped
    
    def __box_center__(self, bbox):
        if bbox is None:
            return None
        center = (bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (bbox[3] - bbox[1]) // 2)
        return center
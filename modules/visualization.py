import numpy as np
import cv2 as cv
import rsp.common.drawing as drawing
import rsp.common.color as colors

class HAR2UnitreeUI():
    def __init__(
            self,
            action_labels
    ):
        self.action_labels = action_labels
        self.framerate = 0.
        self.detected = False
        self.action = 0
        self.scores = np.zeros((len(action_labels),))
        self.confidence = 0.0
        self.buffer_size = 0

    def show(self, img):
        img = cv.cvtColor(np.asarray(img.copy()*255, dtype=np.uint8), cv.COLOR_RGB2BGR) / 255

        foreground = colors.BLACK
        if self.detected:
            background = colors.DARK_GREEN
        else:
            background = colors.DARK_GRAY
        

        img = drawing.add_text(
            img,
            text=f'A{self.action:0>3} - {self.action_labels[self.action]} ({self.confidence:.2f})',
            p=(10, 30),
            foreground=foreground,
            background=background,
            background_opacity=0.7,
            text_thickness=3,
            scale=2.0,
            vertical_align='center',
            width=900,
            margin=5
        )
        img = drawing.add_text(
            img,
            text=f'{self.framerate:.0f} FPS',
            p=(img.shape[1]-250, 30),
            vertical_align='center',
            foreground=colors.BLACK,
            background=colors.WHITE,
            width=240,
            scale=2.0,
            text_thickness=2,
            #height=20,
            background_opacity=0.7
        )
        img = cv.putText(img, f'Buffer size: {self.buffer_size}', (10, 150), cv.FONT_HERSHEY_SIMPLEX, 1, colors.WHITE, 2, cv.LINE_AA)
        img = self.__draw_action_chart__(img, self.scores)

        cv.imshow('HAR2Unitree - UI', img)
        cv.waitKey(1)

    def __draw_action_chart__(self, img, scores):
        px, py = 10, int(0.7 * img.shape[0])#500
        scale = 0.002 * (img.shape[0] - py)
        w, h = img.shape[1] - 2 * px, img.shape[0] - py
        bar_margin = 0
        bar_height = int(h / len(scores))
        for i, score in enumerate(self.scores):
            sx = px 
            sy = py + i * bar_height
            ex = px + w
            ey = sy + bar_height - bar_margin
            img = cv.rectangle(img, (sx, sy), (ex, ey), colors.LIGHT_GRAY, thickness=1)

            bar_width = int(w * score)
            img = drawing.add_text(
                img,
                f'A{i:0>3}',
                (sx, sy),
                width=bar_width,
                height=bar_height-bar_margin,
                foreground=colors.WHITE,
                background=colors.DARK_GREEN,
                background_opacity=0.8,
                scale=scale,
                text_thickness=1,
                margin=2,
                vertical_align='center'
            )
            
        return img
        pass
import numpy as np
import cv2 as cv
import rsp.common.drawing as drawing
import rsp.common.color as colors
from rsp.userinterface import Window, Button
from threading import Thread
from modules.stats import RobotState, ActionLabel
from modules.framework import HAR2UnitreeProcessor
import time

class Robot_UI(Window):
    def __init__(
        self,
        win_name:str,
        func_move_foward:callable,
        func_turn:callable,
        func_joy:callable,
        func_stand_up:callable,
        func_damping:callable,
        size = (500, 300)
    ):
        super().__init__(win_name, size)

        self.func_move_foward = func_move_foward
        self.func_turn = func_turn
        # self.func_joy = func_joy
        # self.func_stand_up = func_stand_up
        # self.func_damping = func_damping

        self.ui_elements = {
            'btn_forward': Button('forward', 45, 10, w=70),#, on_left_button_clicked=lambda: func_move_foward()),
            'btn_turn_left': Button('<', 10, 10, w=30),#, on_left_button_clicked=self.__on_btn_turn_left__),
            'btn_turn_right': Button('>', 120, 10, w=30),#, on_left_button_clicked=self.__on_btn_turn_right__),
            'btn_joy': Button('Process joy', 10, 40, w=150, on_left_button_clicked=lambda: func_joy()),
            'btn_stand_up': Button('Process stand up', 10, 70, w=150, on_left_button_clicked=lambda: func_stand_up()),
            'btn_damping': Button('Process damping', 10, 100, w=150, on_left_button_clicked=lambda: func_damping()),
        }

        for ui_element in self.ui_elements.values():
            self.__ui_elements__.append(ui_element)

        Thread(target=self.__thread_cycle__).start()

    def __thread_cycle__(self):
        while True:
            if self.ui_elements['btn_turn_left'].is_checked:
                self.func_turn(2)
                print('Turn left')
            if self.ui_elements['btn_turn_right'].is_checked:
                self.func_turn(-2)
                print('Turn right')
            if self.ui_elements['btn_forward'].is_checked:
                self.func_move_foward()
                print('Move forward')
            time.sleep(0.2)

class HAR_UI():
    COLOR_DARK_GREEN = (81, 94, 39)
    COLOR_LIGHT_GREEN = (161, 174, 119)
    COLOR_DARK_GRAY = (100, 100, 100)
    COLOR_LIGHT_GRAY = (220, 220, 220)

    OPACITY_LOW = 0.4
    OPACITY_HIGH = 0.8
    TEXT_BOX_HEIGHT = 50
    TEXT_BOX_WIDTH = 400
    MARGIN = 10
    INNER_MARGIN = 5

    def __init__(self, har2unitree_processor:HAR2UnitreeProcessor):
        self.har2unitree_processor = har2unitree_processor

        self.action_labels = [action_label.name.replace('_', ' ') for action_label in ActionLabel]

        self.__img_robot__ = cv.imread('image/unitree_go1.png', cv.IMREAD_UNCHANGED)
        self.__img_robot__ = cv.resize(self.__img_robot__, (0, 0), fx=0.25, fy=0.25)

        # logos
        logo_height = 54
        self.__img_rhmi__ = cv.imread('image/rhmi.png', cv.IMREAD_UNCHANGED)
        f = logo_height / self.__img_rhmi__.shape[0]
        self.__img_rhmi__ = cv.resize(self.__img_rhmi__, (0, 0), fx=f, fy=f)

        logo_height = 70
        self.__img_tuc__ = cv.imread('image/tuc.png', cv.IMREAD_UNCHANGED)
        f = logo_height / self.__img_tuc__.shape[0]
        self.__img_tuc__ = cv.resize(self.__img_tuc__, (0, 0), fx=f, fy=f)

    def show(self):
        if not hasattr(self.har2unitree_processor, 'color_frame'):
            return
        img = cv.cvtColor(np.asarray(self.har2unitree_processor.color_frame.copy()*255, dtype=np.uint8), cv.COLOR_RGB2BGR) / 255
        
        framerate = self.har2unitree_processor.har_processor.framerate
        img = drawing.add_text(
            img,
            text=f'{framerate:.0f} FPS',
            p=(img.shape[1]-210, 10),
            vertical_align='center',
            horizontal_align='right',
            width=200,
            foreground=colors.WHITE,
            scale=1.0,
            text_thickness=2
        )

        img = self.__draw_tracking_results__(img)
        img = self.__draw_robot_state__(img)
        img = self.__draw_current_action__(img, h = self.__img_robot__.shape[0] - HAR_UI.MARGIN, w = self.__img_robot__.shape[1])
        img = self.__draw_actions__(img, px = HAR_UI.MARGIN + self.__img_robot__.shape[1] + HAR_UI.MARGIN)

        bg_p1x, bg_p1y = HAR_UI.MARGIN, HAR_UI.MARGIN
        bg_p2x, bg_p2y = bg_p1x + HAR_UI.INNER_MARGIN + self.__img_rhmi__.shape[1] + HAR_UI.INNER_MARGIN + self.__img_tuc__.shape[1] + HAR_UI.INNER_MARGIN, bg_p1y + HAR_UI.INNER_MARGIN + self.__img_tuc__.shape[0] + HAR_UI.INNER_MARGIN

        tuc_px = bg_p1x + HAR_UI.INNER_MARGIN
        tuc_py = bg_p1y + HAR_UI.INNER_MARGIN

        rhmi_px = tuc_px + self.__img_tuc__.shape[1] + HAR_UI.INNER_MARGIN
        rhmi_py = bg_p1y + HAR_UI.INNER_MARGIN
        
        img = drawing.add_rectangle(img, (bg_p1x, bg_p1y), (bg_p2x, bg_p2y), opacity=HAR_UI.OPACITY_LOW, color=colors.WHITE)
        img = self.__add_transparent_image__(img, self.__img_tuc__, tuc_px, tuc_py)
        img = self.__add_transparent_image__(img, self.__img_rhmi__, rhmi_px, rhmi_py+2)

        cv.imshow('HAR2Unitree - UI', img)
        cv.waitKey(1)

    def __draw_current_action__(self, img, h, w):
        # img = self.__draw_action_chart__(
        #     img, HAR_UI.MARGIN,
        #     img.shape[0] - HAR_UI.MARGIN - h - HAR_UI.TEXT_BOX_HEIGHT - HAR_UI.MARGIN,
        #     w,
        #     h
        # )

        p1x = HAR_UI.MARGIN
        #p1y = img.shape[0] - h - margin
        p1y = img.shape[0] - HAR_UI.MARGIN - HAR_UI.TEXT_BOX_HEIGHT

        # states
        robot_progress = self.har2unitree_processor.go1.progress
        robot_state = self.har2unitree_processor.go1.state
        har_action = self.har2unitree_processor.har_processor.action
        har_action_detected = self.har2unitree_processor.har_processor.detected

        if robot_progress is not None and robot_state != RobotState.TRACKING:
            return img
        elif robot_state == RobotState.TRACKING and har_action != ActionLabel.Stop.value:
            background = colors.LIGHT_GRAY
        elif har_action_detected:
            background = HAR_UI.COLOR_DARK_GREEN
        else:
            background = colors.DARK_RED

        img = self.__draw_text_block__(
            img,
            #f'A{self.action:0>3} - {self.action_labels[self.action]} ({self.confidence:0.2f})',
            f'{self.action_labels[har_action]}',
            background=background,
            foreground=colors.WHITE,
            x=p1x,
            y=p1y,
            w=w,
            h=HAR_UI.TEXT_BOX_HEIGHT
        )

        return img
    
    def __draw_text_block__(self, img, text, background, foreground, x, y, w, h):
        img = cv.circle(img, (x + h//2, y + h//2), h//2, background, -1)
        img = cv.circle(img, (x + w - h//2, y + h//2), h//2, background, -1)
        img = cv.rectangle(img, (x + h//2, y), (x + w - h//2, y+h), background, -1)

        img = drawing.add_text(img, text,
            (x, y),
            width=w,
            height=h,
            foreground=foreground,
            scale=1.0,
            text_thickness=2,
            margin=2,
            vertical_align='center',
            horizontal_align='center'
        )
        return img

    def __draw_robot_state__(self, img):
        # robot state
        robot_state = self.har2unitree_processor.go1.state
        px = img.shape[1]-self.__img_robot__.shape[1] - HAR_UI.MARGIN
        img = self.__draw_text_block__(
            img,
            f'{RobotState(robot_state).name.replace("_", " ")}',
            background=HAR_UI.COLOR_DARK_GREEN,
            foreground=colors.WHITE,
            x=px,
            y=img.shape[0] - HAR_UI.MARGIN - HAR_UI.TEXT_BOX_HEIGHT,
            w=self.__img_robot__.shape[1],
            h=HAR_UI.TEXT_BOX_HEIGHT
        )
        img = self.__add_transparent_image__(img, self.__img_robot__,
            x_offset=px,
            y_offset=img.shape[0]-self.__img_robot__.shape[0] - HAR_UI.MARGIN - HAR_UI.TEXT_BOX_HEIGHT)
        
        # robot progress
        robot_progress = self.har2unitree_processor.go1.progress
        h = 10
        w = self.__img_robot__.shape[1]
        p1x = px
        p1y = img.shape[0] - HAR_UI.MARGIN - self.__img_robot__.shape[0]-HAR_UI.TEXT_BOX_HEIGHT-h
        p2x = img.shape[1] - HAR_UI.MARGIN
        p2y = p1y+h

        if robot_progress is not None and robot_state != RobotState.TRACKING:
            img = drawing.add_rectangle(img,
                (p1x, p1y),
                (p2x, p2y),
                opacity=HAR_UI.OPACITY_LOW,
                color=colors.WHITE
            )
            ex = int(px + robot_progress * w)
            img = drawing.add_rectangle(img,
                (px, p1y),
                (ex, p2y),
                opacity=HAR_UI.OPACITY_HIGH,
                color=HAR_UI.COLOR_DARK_GREEN
            )
            img = cv.rectangle(img, (p1x, p1y), (p2x, p2y), colors.WHITE, thickness=1)
        
        return img

    def __draw_action_chart__(self, img, px, py, w, h):
        har_scores = self.har2unitree_processor.har_processor.scores
        scale = 0.002 * (img.shape[0] - py)
        bar_margin = 0
        bar_height = int(h / len(har_scores))
        for i, score in enumerate(har_scores):
            sx = px 
            sy = py + i * bar_height
            ex = px + w
            ey = sy + bar_height - bar_margin

            img = drawing.add_rectangle(img, (sx, sy), (ex, ey), color=colors.WHITE, opacity=HAR_UI.OPACITY_LOW)

            bar_width = int(w * score)
            img = drawing.add_text(
                img,
                f'A{i:0>3}',
                (sx, sy),
                width=bar_width,
                height=bar_height-bar_margin,
                foreground=colors.WHITE,
                background=HAR_UI.COLOR_DARK_GREEN,
                background_opacity=1.,
                scale=scale,
                text_thickness=1,
                margin=2,
                vertical_align='center'
            )
            img = drawing.add_text(
                img,
                f'{score:0.2f}',
                (sx, sy),
                width=w,
                height=bar_height-bar_margin,
                foreground=colors.WHITE,
                scale=scale,
                text_thickness=1,
                margin=2,
                vertical_align='center',
                horizontal_align='right'
            )
            img = cv.rectangle(img, (sx, sy), (ex, ey), colors.WHITE, thickness=1)

        return img

    def __draw_actions__(self, img, px):
        robot_state = self.har2unitree_processor.go1.state
        next_actions = HAR2UnitreeProcessor.NEXT_ACTIONS[robot_state]

        def draw_action(img, cx, cy, action, label, score, active, r, detected, w = 220):
            if active:
                background = HAR_UI.COLOR_LIGHT_GREEN
                foreground = colors.CORNFLOWER_BLUE if detected else HAR_UI.COLOR_DARK_GREEN
                text_color = colors.CORNFLOWER_BLUE if detected else HAR_UI.COLOR_DARK_GREEN
            else:
                background = HAR_UI.COLOR_LIGHT_GRAY
                foreground = HAR_UI.COLOR_DARK_GRAY
                text_color = HAR_UI.COLOR_LIGHT_GRAY if detected else HAR_UI.COLOR_DARK_GRAY

            
            img = drawing.add_rectangle(
                img,
                (cx - r - HAR_UI.INNER_MARGIN // 2, cy - r - HAR_UI.INNER_MARGIN // 2),
                (cx + w - r - HAR_UI.INNER_MARGIN // 2, cy + r + HAR_UI.INNER_MARGIN // 2),
                opacity=HAR_UI.OPACITY_LOW,
                color=colors.WHITE
            )

            img = cv.circle(img, (cx, cy), radius=r, color=foreground, thickness=-1)
            end_angle = -90 + 360 * score
            if end_angle > 0.002:
                thickness = 0
                img = cv.ellipse(img, (cx, cy), (r - thickness//2, r-thickness//2), 0, -90, end_angle, background, thickness=-1)
            img = drawing.add_text(
                img,
                f'{label}',
                (cx + r + HAR_UI.INNER_MARGIN, cy-r),
                # width=2*r,
                height=2*r,
                foreground=text_color,
                scale=1.0,
                text_thickness=2,
                margin=2,
                vertical_align='center',
                horizontal_align='left'
            )
            img = drawing.add_text(
                img,
                f'{score:0.3f}',
                (cx, cy-r//2),
                width=w - r - HAR_UI.INNER_MARGIN // 2,
                height= r,
                foreground=text_color,
                scale=0.5,
                text_thickness=1,
                margin=4,
                vertical_align='center',
                horizontal_align='right'
            )
            return img

        r = 20
        sx, sy = HAR_UI.MARGIN + HAR_UI.INNER_MARGIN // 2, 220
        for action, label in enumerate(self.action_labels):
            score = self.har2unitree_processor.har_processor.scores[action].item()
            detected = self.har2unitree_processor.har_processor.detected and self.har2unitree_processor.har_processor.action == action

            img = draw_action(
                img,
                sx + r,
                sy + action * (2 * r + HAR_UI.MARGIN),
                action,
                label,
                score,
                ActionLabel(action) in next_actions,
                r,
                detected,
                w = self.INNER_MARGIN + self.__img_tuc__.shape[1] + HAR_UI.INNER_MARGIN + self.__img_rhmi__.shape[1] + HAR_UI.INNER_MARGIN
            )
            # cx = px + action * (2 * r + HAR_UI.MARGIN) + r
            # if ActionLabel(action) in next_actions:
            #     background = HAR_UI.COLOR_LIGHT_GREEN
            #     foreground = HAR_UI.COLOR_DARK_GREEN
            # else:
            #     background = HAR_UI.COLOR_LIGHT_GRAY
            #     foreground = HAR_UI.COLOR_DARK_GRAY


            # img = cv.circle(img, (cx, cy), radius=r, color=background, thickness=-1)

            # end_angle = -90 + 360 * score
            # if end_angle > 0.002:
            #     img = cv.ellipse(img, (cx, cy), (r, r), 0, -90, end_angle, foreground, thickness=5)

            # img = drawing.add_text(
            #     img,
            #     f'{label}',
            #     (cx, cy),
            #     # width=2*r,
            #     # height=2*r,
            #     foreground=foreground,
            #     scale=0.5,
            #     text_thickness=1,
            #     margin=2,
            #     vertical_align='center',
            #     horizontal_align='center'
            # )

        return img

    def __draw_tracking_results__(self, img):
        tracking_results = self.har2unitree_processor.tracking_results
        for result in tracking_results:
            bbox = result['bbox']
            s_pos_max = result['s_pos_max']
            s_neg_max = result['s_neg_max']
            detected = result['detected']
            if detected:
                color = colors.CORNFLOWER_BLUE
            else:
                color = colors.DARK_RED
            img = cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=2)
            #img = cv.putText(img, f'{s_pos_max:.2f}', (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
            img = cv.putText(img, f'{s_pos_max - s_neg_max:.2f}', (bbox[0], bbox[1]+20), cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv.LINE_AA)
        return img
    
    def __add_transparent_image__(self, background, foreground, x_offset=None, y_offset=None):
        bg_h, bg_w, bg_channels = background.shape
        fg_h, fg_w, fg_channels = foreground.shape

        assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
        assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

        # center by default
        if x_offset is None: x_offset = (bg_w - fg_w) // 2
        if y_offset is None: y_offset = (bg_h - fg_h) // 2

        w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
        h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

        if w < 1 or h < 1: return

        # clip foreground and background images to the overlapping regions
        bg_x = max(0, x_offset)
        bg_y = max(0, y_offset)
        fg_x = max(0, x_offset * -1)
        fg_y = max(0, y_offset * -1)
        foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
        background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

        # separate alpha and color channels from the foreground image
        foreground_colors = foreground[:, :, :3]
        alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

        # construct an alpha_mask that matches the image shape
        alpha_mask = alpha_channel[:, :, np.newaxis]

        # combine the background with the overlay image weighted by alpha
        composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

        # overwrite the section of the background image that has
        background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

        return background

if __name__ == '__main__':
    robot_ui = Robot_UI(
        win_name='Robot UI',
        func_turn=lambda angle: None,#print(f'Turn {angle:0.2f}'),
        func_joy=lambda: None#print(f'Joy')
    )

    while True:
        robot_ui.__render__()
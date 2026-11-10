import cv2
import numpy as np
import pyautogui
import math
import time
import win32api
import win32con
import customtkinter as ctk
from PIL import Image, ImageTk
from collections import deque

import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
pyautogui.FAILSAFE = False

class SmoothingBuffer:
    """Temporal filtering for smooth gesture tracking"""
    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, value):
        self.buffer.append(value)
        return self.get_average()
    
    def get_average(self):
        if len(self.buffer) == 0:
            return None
        return tuple(np.mean(list(self.buffer), axis=0))

class GestureEngine:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.85,  # ⬆️ Higher confidence = more stable
            min_tracking_confidence=0.8     # ⬆️ Better tracking
        )
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.canvas = None
        self.draw_color = (0, 255, 255)
        self.px, self.py = 0, 0
        self.current_mode = "INITIALIZING..."
        
        # ===== IMPROVED SMOOTHING =====
        self.mouse_smoother = SmoothingBuffer(max_size=3)
        self.plocX, self.plocY = 0, 0
        
        # ===== STRONGER GESTURE STABILITY =====
        self.last_gesture = None
        self.gesture_stability_count = 0
        self.min_stability_frames = 5  # ⬆️ INCREASED: Need 5 stable frames (was 2)
        
        # ===== FINGER STATE HISTORY (NEW) =====
        self.finger_history = deque(maxlen=3)  # Keep last 3 finger detections
        
        self.last_pinch_time = 0
        self.pinch_state = False
        
        self.VK_VOLUME_DOWN = 0xAE
        self.VK_VOLUME_UP = 0xAF
        self.prev_vol_dist = None
        self.volume_history = deque(maxlen=2)
        
        self.roi_top, self.roi_bottom = 0.2, 0.7
        self.roi_left, self.roi_right = 0.2, 0.8

    def get_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def get_finger_states_strict(self, lm):
        """IMPROVED: Stricter finger detection with hysteresis"""
        wrist = lm[0]
        tips, mcps = [8, 12, 16, 20], [5, 9, 13, 17]
        fingers = []
        
        for tip, mcp in zip(tips, mcps):
            tip_dist = self.get_distance(lm[tip], wrist)
            mcp_dist = self.get_distance(lm[mcp], wrist)
            
            # ⬇️ IMPORTANT: Add margin of 5px for hysteresis (prevents jitter)
            if tip_dist > (mcp_dist + 5):  
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers

    def get_stable_finger_state(self, current_fingers):
        """
        NEW: Check if finger state is consistent across multiple frames
        This prevents flickering between gestures
        """
        self.finger_history.append(tuple(current_fingers))
        
        if len(self.finger_history) < 2:
            return current_fingers
        
        # Check if all recent finger states are the same
        if all(state == self.finger_history[0] for state in self.finger_history):
            return list(self.finger_history[0])
        
        # If not stable, return LAST stable state (don't jump around)
        return list(self.finger_history[0])

    def classify_gesture(self, stable_fingers, index_tip, thumb_tip, middle_tip):
        """Classify with STRICTER matching"""
        
        pinch_dist = self.get_distance(index_tip, middle_tip)
        
        # ===== EXACT MATCHING (NO WILD CARDS) =====
        if stable_fingers == [1, 1, 1, 0]:
            gesture = "AIR_CANVAS"
        elif stable_fingers == [1, 1, 0, 0]:
            gesture = "VIRTUAL_MOUSE"
        elif stable_fingers == [1, 0, 0, 0]:
            gesture = "VOLUME_CONTROL"
        elif stable_fingers == [0, 0, 0, 0]:
            gesture = "CLEAR"
        elif stable_fingers == [1, 1, 1, 1]:
            gesture = "CLEAR"
        else:
            # ⬇️ FALLBACK: Keep last gesture if ambiguous
            gesture = self.last_gesture if self.last_gesture else "RESTING"
        
        # ===== ULTRA-STRICT STABILITY REQUIREMENT =====
        if gesture == self.last_gesture:
            self.gesture_stability_count += 1
        else:
            self.gesture_stability_count = 1
        
        # ⬇️ NEED 5 CONSECUTIVE STABLE FRAMES TO SWITCH
        if self.gesture_stability_count >= self.min_stability_frames:
            current_stable_gesture = gesture
        else:
            # Stay with previous gesture until 5 frames pass
            current_stable_gesture = self.last_gesture or "RESTING"
        
        self.last_gesture = gesture
        
        return current_stable_gesture, pinch_dist

    def process_frame(self):
        success, frame = self.cap.read()
        if not success or frame is None:
            return None

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        roi_top = int(h * self.roi_top)
        roi_bottom = int(h * self.roi_bottom)
        roi_left = int(w * self.roi_left)
        roi_right = int(w * self.roi_right)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        self.current_mode = "RESTING 😴"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                lm = []
                for pt in hand_landmarks.landmark:
                    cx, cy = int(pt.x * w), int(pt.y * h)
                    lm.append((cx, cy))

                if lm:
                    # ===== IMPROVED FINGER DETECTION =====
                    raw_fingers = self.get_finger_states_strict(lm)
                    stable_fingers = self.get_stable_finger_state(raw_fingers)
                    
                    index_tip, thumb_tip, middle_tip = lm[8], lm[4], lm[12]
                    
                    # Get stable gesture
                    gesture, pinch_dist = self.classify_gesture(stable_fingers, index_tip, thumb_tip, middle_tip)

                    # ===== MODE 1: AIR CANVAS =====
                    if gesture == "AIR_CANVAS":
                        self.current_mode = "AIR CANVAS 🎨"
                        ix, iy = index_tip[0], index_tip[1]
                        
                        if self.px == 0 and self.py == 0:
                            self.px, self.py = ix, iy
                        else:
                            cv2.line(self.canvas, (self.px, self.py), (ix, iy), self.draw_color, 6)
                        
                        self.px, self.py = ix, iy
                        self.prev_vol_dist = None

                    # ===== MODE 2: VIRTUAL MOUSE =====
                    elif gesture == "VIRTUAL_MOUSE":
                        self.current_mode = "VIRTUAL MOUSE 🖱️"
                        self.px, self.py = 0, 0
                        ix, iy = index_tip[0], index_tip[1]
                        
                        x3 = np.interp(ix, (roi_left, roi_right), (0, self.screen_w))
                        y3 = np.interp(iy, (roi_top, roi_bottom), (0, self.screen_h))

                        smoothed_pos = self.mouse_smoother.add((x3, y3))
                        if smoothed_pos:
                            x3, y3 = smoothed_pos

                        current_time = time.time()

                        if pinch_dist < 38:
                            if not self.pinch_state:
                                self.pinch_state = True
                                
                                if (current_time - self.last_pinch_time) < 0.5:
                                    pyautogui.doubleClick()
                                    cv2.putText(frame, "DOUBLE CLICK!", (ix, iy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                    self.last_pinch_time = 0
                                else:
                                    pyautogui.click()
                                    cv2.putText(frame, "CLICK!", (ix, iy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                                    self.last_pinch_time = current_time
                        else:
                            self.pinch_state = False
                            pyautogui.moveTo(int(x3), int(y3))

                        self.prev_vol_dist = None

                    # ===== MODE 3: VOLUME CONTROL =====
                    elif gesture == "VOLUME_CONTROL":
                        self.current_mode = "VOLUME CONTROL 🔊"
                        self.px, self.py = 0, 0
                        
                        cv2.circle(frame, thumb_tip, 8, (255, 0, 255), cv2.FILLED)
                        cv2.circle(frame, index_tip, 8, (255, 0, 255), cv2.FILLED)
                        cv2.line(frame, thumb_tip, index_tip, (255, 0, 255), 2)

                        dist = self.get_distance(index_tip, thumb_tip)
                        self.volume_history.append(dist)

                        if len(self.volume_history) == 2:
                            prev_dist = self.volume_history[0]
                            diff = dist - prev_dist
                            
                            if abs(diff) > 5:
                                if diff > 0:
                                    win32api.keybd_event(self.VK_VOLUME_UP, 0, 0, 0)
                                    win32api.keybd_event(self.VK_VOLUME_UP, 0, win32con.KEYEVENTF_KEYUP, 0)
                                else:
                                    win32api.keybd_event(self.VK_VOLUME_DOWN, 0, 0, 0)
                                    win32api.keybd_event(self.VK_VOLUME_DOWN, 0, win32con.KEYEVENTF_KEYUP, 0)

                    # ===== MODE 4: CLEAR =====
                    elif gesture == "CLEAR":
                        self.current_mode = "CLEARING CANVAS 🧹"
                        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        self.px, self.py = 0, 0
                        self.prev_vol_dist = None
                    
                    else:
                        self.px, self.py = 0, 0
                        self.prev_vol_dist = None

        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 255), 1)
        frame = cv2.add(frame, self.canvas)

        return frame

    def close_cam(self):
        self.cap.release()

# ========== DASHBOARD GUI ==========
class AppGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("VisionTouch AI - Spatial Interface Dashboard")
        self.geometry("1000x650")
        self.resizable(False, False)

        self.engine = GestureEngine()
        self.prev_time = time.time()

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # SIDEBAR
        self.sidebar = ctk.CTkFrame(self, corner_radius=15)
        self.sidebar.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar, text="VisionTouch AI", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(padx=20, pady=(20, 10))

        self.subtitle_label = ctk.CTkLabel(self.sidebar, text="Spatial Control Engine", font=ctk.CTkFont(size=11), text_color="gray")
        self.subtitle_label.pack(padx=20, pady=(0, 15))

        self.mode_card = ctk.CTkFrame(self.sidebar, fg_color="#1E1E2E", corner_radius=10)
        self.mode_card.pack(padx=15, pady=8, fill="x")

        self.mode_title = ctk.CTkLabel(self.mode_card, text="ACTIVE MODE", font=ctk.CTkFont(size=10, weight="bold"), text_color="#89B4FA")
        self.mode_title.pack(padx=10, pady=(6, 2))

        self.mode_val = ctk.CTkLabel(self.mode_card, text="STARTING...", font=ctk.CTkFont(size=13, weight="bold"))
        self.mode_val.pack(padx=10, pady=(0, 6))

        self.fps_card = ctk.CTkFrame(self.sidebar, fg_color="#1E1E2E", corner_radius=10)
        self.fps_card.pack(padx=15, pady=8, fill="x")

        self.fps_title = ctk.CTkLabel(self.fps_card, text="PERFORMANCE", font=ctk.CTkFont(size=10, weight="bold"), text_color="#A6E3A1")
        self.fps_title.pack(padx=10, pady=(6, 2))

        self.fps_val = ctk.CTkLabel(self.fps_card, text="0 FPS", font=ctk.CTkFont(size=13, weight="bold"))
        self.fps_val.pack(padx=10, pady=(0, 6))

        self.slider_label = ctk.CTkLabel(self.sidebar, text="Mouse Smoothing", font=ctk.CTkFont(size=11))
        self.slider_label.pack(padx=20, pady=(20, 2))

        self.sens_slider = ctk.CTkSlider(self.sidebar, from_=2, to=5, number_of_steps=3, command=self.update_smoothing)
        self.sens_slider.set(3)
        self.sens_slider.pack(padx=20, pady=(0, 15))

        self.quit_btn = ctk.CTkButton(self.sidebar, text="Exit Application", fg_color="#F38BA8", hover_color="#D20F39", command=self.on_close)
        self.quit_btn.pack(padx=20, pady=(15, 15), side="bottom")

        # MAIN PANEL
        self.main_panel = ctk.CTkFrame(self, corner_radius=15, fg_color="#181825")
        self.main_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.main_panel, text="")
        self.video_label.pack(expand=True, padx=15, pady=15)

        self.update_video()

    def update_smoothing(self, value):
        self.engine.mouse_smoother = SmoothingBuffer(max_size=int(value))

    def on_close(self):
        self.engine.close_cam()
        self.destroy()

    def update_video(self):
        frame = self.engine.process_frame()

        if frame is not None:
            curr_time = time.time()
            fps = int(1 / (curr_time - self.prev_time + 1e-5))
            self.prev_time = curr_time

            self.mode_val.configure(text=self.engine.current_mode)
            self.fps_val.configure(text=f"{fps} FPS")

            cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_img)
            img = img.resize((620, 460), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(10, self.update_video)

if __name__ == "__main__":
    app = AppGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
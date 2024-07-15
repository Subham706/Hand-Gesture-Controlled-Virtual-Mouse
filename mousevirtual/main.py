import cv2
import mediapipe as mp
import pyautogui
import logging
import os

# Suppress TensorFlow and other library warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

# Disable PyAutoGUI fail-safe (Use with caution)
pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
scroll_threshold = 30  # Threshold for initiating scroll action

index_y = None  # Initialize index finger y-position
middle_y = None  # Initialize middle finger y-position

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:  # Index finger tip
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    index_y = y

                if id == 12:  # Middle finger tip
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(255, 0, 0))
                    middle_y = y

            # Scroll logic
            if index_y is not None and middle_y is not None:
                y_diff = index_y - middle_y
                if abs(y_diff) > scroll_threshold:
                    if y_diff > 0:
                        print("Scrolling up")  # Debugging print
                        pyautogui.scroll(5)  # Scroll up
                    else:
                        print("Scrolling down")  # Debugging print
                        pyautogui.scroll(-5)  # Scroll down

    cv2.imshow('virtualmouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

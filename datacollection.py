import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector
import sys
from PIL import Image, ImageDraw, ImageFont

# Ensure UTF-8 encoding for Kannada text
sys.stdout.reconfigure(encoding='utf-8')

# Initialize camera and hand detector
cam = cv2.VideoCapture(0)
hand_find = HandDetector(maxHands=1)

# Image settings
img_dim = 300
marg = 20
save_directory = "Data/Ee"
img_count = 0

# Reference values for distance estimation
KNOWN_WIDTH = 10  # Approximate width of a hand in cm
FOCAL_LENGTH = 600  # Focal length (estimated, needs calibration)

# Load Kannada font (Ensure font is available)
try:
    font = ImageFont.truetype("NotoSansKannada.ttf", 40)
except IOError:
    print("Kannada font not found! Using default font.")
    font = ImageFont.load_default()

def estimate_distance(perceived_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width

while True:
    success, frame = cam.read()
    if not success:
        continue

    output_frame = frame.copy()
    detected_hands, frame = hand_find.findHands(frame, draw=True)

    if detected_hands:
        hand_info = detected_hands[0]
        a, b, width, height = hand_info['bbox']
        crop_hands = frame[b - marg:b + height + marg, a - marg:a + width + marg]

        # Estimate distance based on bounding box width
        distance_cm = estimate_distance(width)
        distance_text = f"Distance: {int(distance_cm)} cm"

        # Display distance warning
        if distance_cm < 30:
            cv2.putText(output_frame, "Move Hands further!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif 30 <= distance_cm <= 40:
            cv2.putText(output_frame, "Good Position!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(output_frame, "Come Closer", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

        cv2.putText(output_frame, distance_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw hand bounding box
        cv2.rectangle(output_frame, (a - marg, b - marg), (a + width + marg, b + height + marg), (255, 0, 255), 4)

        # Draw hand landmarks (key points)
        for lm in hand_info['lmList']:
            x, y = lm[:2]  # Extract x, y coordinates
            cv2.circle(output_frame, (x, y), 5, (0, 255, 0), -1)  # Draw green dots

        # Draw connections between landmarks (optional, for better visualization)
        connections = hand_info['lmList']
        for i in range(len(connections) - 1):
            cv2.line(output_frame, connections[i][:2], connections[i + 1][:2], (255, 0, 0), 2)

    cv2.imshow("Live Feed", output_frame)
    key_press = cv2.waitKey(1)

    if key_press == ord("s"):
        img_count += 1
        cv2.imwrite(f'{save_directory}/Captured_{time.time()}.jpg', frame)
        print(f"Saved Image: {img_count}")

    if key_press == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
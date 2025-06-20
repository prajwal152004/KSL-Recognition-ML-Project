import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import sys
from PIL import Image, ImageDraw, ImageFont

# Ensure UTF-8 encoding for Kannada text
sys.stdout.reconfigure(encoding='utf-8')

# Initialize camera, hand detector, and classifier
cam = cv2.VideoCapture(0)
hand_find = HandDetector(maxHands=1)
predict = Classifier("mod/keras_model.h5", "mod/labels.txt")

# Image settings
img_dim = 300
marg = 20

# Kannada label mapping
label_map = {0: "ಅ", 1: "ಆ", 2: "ಇ",3:"ಈ"}

# Load Kannada font (Ensure font is available)
try:
    font = ImageFont.truetype("NotoSansKannada.ttf", 40)
except IOError:
    print("Kannada font not found! Using default font.")
    font = ImageFont.load_default()


def estimate_distance(perceived_width):
    KNOWN_WIDTH = 10  # Approximate width of a hand in cm
    FOCAL_LENGTH = 600  # Estimated focal length
    return (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width


while True:
    success, frame = cam.read()
    if not success:
        continue

    output_frame = frame.copy()
    detected_hands, frame = hand_find.findHands(frame)

    if detected_hands:
        hand_info = detected_hands[0]
        a, b, width, height = hand_info['bbox']
        crop_hands = frame[b - marg:b + height + marg, a - marg:a + width + marg]

        distance_cm = estimate_distance(width)
        distance_text = f"Distance: {int(distance_cm)} cm"

        # Display distance warnings
        if distance_cm < 30:
            cv2.putText(output_frame, "Move Hands further!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif 30 <= distance_cm <= 40:
            cv2.putText(output_frame, "Good Position!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(output_frame, "Come Closer", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

        cv2.putText(output_frame, distance_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if crop_hands.size != 0:
            prediction, index = predict.getPrediction(crop_hands, draw=False)
            detected_char = label_map.get(index, "?")

            # Convert OpenCV image to PIL format
            pil_img = Image.fromarray(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # Draw a separate box above the hand box for Kannada text
            text_box_top = b - marg - 50  # Position above the hand box
            text_box_bottom = b - marg - 10
            text_box_left = a - marg
            text_box_right = a + width + marg

            draw.rectangle([text_box_left, text_box_top, text_box_right, text_box_bottom], fill=(0, 0, 0))
            draw.text((a + width // 2 - 20, text_box_top + 5), detected_char, font=font, fill=(255, 255, 255))

            # Convert back to OpenCV format
            output_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cv2.rectangle(output_frame, (a - marg, b - marg), (a + width + marg, b + height + marg), (255, 0, 255), 4)

    cv2.imshow("Live Prediction", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

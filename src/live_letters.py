"""
    python live_letters.py

Controls:
    SPACE - insert a space
    B     - backspace
    C     - clear sentence
    ESC   - quit
"""

import json
import os

import cv2
import numpy as np
import torch
import mediapipe as mp

from landmarks import make_hands_detector, extract_features, TOTAL_FEATURES
from model import LetterMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "../models"

LETTER_CONFIDENCE_MIN = 0.6
STATIC_FRAMES_NEEDED = 5
LETTER_COOLDOWN = 15


def load_letter_model():
    with open(os.path.join(MODELS_DIR, "letter_labels.json")) as f:
        classes = json.load(f)
    model = LetterMLP(input_dim=TOTAL_FEATURES, num_classes=len(classes))
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "letter_model.pt"), map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, classes


@torch.no_grad()
def predict_letter(model, classes, feats):
    x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
    out = torch.softmax(model(x), dim=1)
    conf, idx = out.max(dim=1)
    return classes[idx.item()], conf.item()


def main():
    model, classes = load_letter_model()
    print(f"Loaded letter model with classes: {classes}")

    cap = cv2.VideoCapture(0)
    hands = make_hands_detector(static_mode=False)

    sentence = ""
    still_count = 0
    cooldown = 0
    last_text = ""

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display = frame.copy()
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    display, hand_lm, mp.solutions.hands.HAND_CONNECTIONS
                )

        feats = extract_features(results)
        hand_present = feats.any()

        if cooldown > 0:
            cooldown -= 1

        top_pred_text = ""
        if hand_present:
            still_count += 1
            letter, conf = predict_letter(model, classes, feats)
            top_pred_text = f"Seeing: {letter} ({conf:.2f})"

            if still_count >= STATIC_FRAMES_NEEDED and cooldown == 0 and conf >= LETTER_CONFIDENCE_MIN:
                sentence += letter
                last_text = f"LOCKED IN: {letter} ({conf:.2f})"
                cooldown = LETTER_COOLDOWN
                still_count = 0
        else:
            still_count = 0

        h, w, _ = display.shape
        cv2.rectangle(display, (0, h - 110), (w, h), (0, 0, 0), -1)
        cv2.putText(display, top_pred_text, (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(display, last_text, (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(display, sentence[-60:], (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Letter Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            sentence += " "
        elif key in (ord("b"), ord("B")):
            sentence = sentence[:-1]
        elif key in (ord("c"), ord("C")):
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
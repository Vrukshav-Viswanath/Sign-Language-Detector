
#Every other script (data collection, training, live inference) imports
#from here so the feature representation is always identical.

#If you change the math here, you need to retrain. 


import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

NUM_LANDMARKS = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3  # x, y, z
MAX_HANDS = 2
TOTAL_FEATURES = FEATURES_PER_HAND * MAX_HANDS  # 126, supports two-handed signs


def make_hands_detector(static_mode: bool = False):
    return mp_hands.Hands(
        static_image_mode=static_mode,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )


def normalize_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
    wrist = landmarks_xyz[0]
    centered = landmarks_xyz - wrist
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).flatten()


def extract_features(results) -> np.ndarray:
    features = np.zeros(TOTAL_FEATURES, dtype=np.float32)

    if not results.multi_hand_landmarks:
        return features

    # Sort detected hands so Left always fills slot 0, Right fills slot 1.
    hands_data = []
    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks, results.multi_handedness
    ):
        label = handedness.classification[0].label  # Left or Right
        xyz = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        hands_data.append((label, normalize_landmarks(xyz)))

    hands_data.sort(key=lambda t: t[0])  # Left < Right alphabetically

    for i, (_, feat) in enumerate(hands_data[:MAX_HANDS]):
        features[i * FEATURES_PER_HAND : (i + 1) * FEATURES_PER_HAND] = feat

    return features


def hand_movement(prev_features: np.ndarray, curr_features: np.ndarray) -> float:
    if prev_features is None:
        return 0.0
    return float(np.linalg.norm(curr_features - prev_features))

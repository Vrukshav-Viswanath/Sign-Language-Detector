"""
Convert the Kaggle "ASL Alphabet" image dataset into landmark .npy files
that train.py can consume directly.

Setup:
    1. Download the dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
       (or `kaggle datasets download -d grassknoted/asl-alphabet` if you have
       the Kaggle CLI + API key configured)
    2. Unzip it. You'll get a folder like:
           asl_alphabet_train/
               A/  (3000 images)
               B/
               ...
               Z/
               space/  del/  nothing/
    3. Run:
        python prepare_letters_dataset.py --source /path/to/asl_alphabet_train

By default this samples SAMPLES_PER_CLASS images per letter (you don't need
all 3000, a few hundred with good hand detection is plenty and much faster).
"""

import argparse
import os
import random

import cv2
import numpy as np

from landmarks import make_hands_detector, extract_features

SAMPLES_PER_CLASS = 400


def process_class(class_dir: str, out_dir: str, hands, samples_per_class: int):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)

    saved = 0
    for fname in files:
        if saved >= samples_per_class:
            break
        img = cv2.imread(os.path.join(class_dir, fname))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        feats = extract_features(results)
        if not feats.any():
            continue  # no hand detected, skip
        np.save(os.path.join(out_dir, f"{saved}.npy"), feats)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                         help="Path to the unzipped asl_alphabet_train folder")
    parser.add_argument("--out", default="../data/letters")
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument("--skip-classes", nargs="*", default=["space", "del", "nothing"],
                         help="Class folders to skip (default: the non-letter utility classes)")
    args = parser.parse_args()

    hands = make_hands_detector(static_mode=True)  # static_mode=True: better for isolated images

    class_dirs = sorted(
        d for d in os.listdir(args.source)
        if os.path.isdir(os.path.join(args.source, d)) and d not in args.skip_classes
    )
    print(f"Found {len(class_dirs)} classes: {class_dirs}")

    total = 0
    for cls in class_dirs:
        src = os.path.join(args.source, cls)
        dst = os.path.join(args.out, cls.upper())
        n = process_class(src, dst, hands, args.samples_per_class)
        total += n
        print(f"  {cls}: saved {n} landmark samples")

    print(f"Done. {total} total samples across {len(class_dirs)} classes -> {args.out}")
    print("Next: python train.py --mode letter")


if __name__ == "__main__":
    main()
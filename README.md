# American Sign Language Alphabet Detector

A real-time webcam app that recognizes American Sign Language letters using MediaPipe hand landmarks and a PyTorch MLP.

Instead of classifying raw video frames, every frame is reduced to a 126-number hand landmark vector (21 points × x/y/z × up to 2 hands) via MediaPipe. This feeds into a small MLP that classifies which letter the hand shape represents.

Trained on the [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (~10,400 samples across 26 classes) — **98.2% validation accuracy**.

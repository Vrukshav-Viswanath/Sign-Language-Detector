# American Sign Language Alphabet Detector
 
A real-time American Sign Language (ASL) alphabet recognizer built with MediaPipe and PyTorch. The model classifies static hand poses into ASL letters using hand landmark features rather than raw pixels, making it lightweight and fast enough to run live from a webcam.
 
## Overview
 
This project detects and classifies the 26 letters of the ASL alphabet from hand gestures. Instead of feeding raw images into a CNN, it uses **MediaPipe Hands** to extract 21 3D hand landmarks per frame, converts them into a normalized 126-dimensional feature vector, and passes that into a **PyTorch MLP** classifier.
 
This landmark-based approach is more robust to lighting, background, and skin tone variation than raw image classification, and trains significantly faster.

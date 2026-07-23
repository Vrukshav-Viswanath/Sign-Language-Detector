# Live ASL Letter Detector
 
A real-time webcam app that recognizes American Sign Language letters (A-Y, excluding J and Z, which involve motion rather than a static hand shape) using MediaPipe hand landmarks and a PyTorch MLP.
 
Instead of classifying raw video frames, every frame is reduced to a 126-number hand landmark vector (21 points x/y/z x up to 2 hands) via MediaPipe. This feeds into a small MLP that classifies which letter the hand shape represents.
 
Trained on the [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (~10,400 samples across 26 classes). Achieved a **98.2% validation accuracy**.
 
## Setup
 
1. Python 3.10 or 3.11 (MediaPipe support is inconsistent on newer versions).
2. Install PyTorch with CUDA support for your GPU (Adjust the CUDA version to match your driver. This project was built against CUDA 12.8 for an RTX 50-series GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
 
(If you don't have a GPU, just `pip install torch` instead.)
 
3. Install the rest:
```bash
pip install -r requirements.txt
```
 
4. Run everything from inside `src/`. All the scripts use relative paths (`../data`, `../models`).
## Project structure
 
```
asl-letter-detector/
├── data/                        # landmark .npy files (gitignored, regenerate below)
│   └── letters/<LETTER>/*.npy
├── models/                      # trained checkpoint + label map (gitignored)
├── src/
│   ├── landmarks.py               # shared MediaPipe feature extraction
│   ├── model.py                   # LetterMLP architecture
│   ├── prepare_letters_dataset.py # convert Kaggle ASL Alphabet to landmarks
│   ├── train.py                   # trains the model
│   └── live_letters.py            # live letter recognition
├── requirements.txt
└── README.md
```
 
## Reproducing it
 
1. Download the [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and unzip it.
2. Convert it into landmark features:
```bash
python prepare_letters_dataset.py --source /path/to/asl_alphabet_train
```
 
3. Train:
```bash
python train.py --mode letter
```
 
4. Run live:
```bash
python live_letters.py
```
 
## Notes
 
- Letters are classified from a single still frame. Hold the hand shape steady for it to register.
- J and Z are intentionally excluded, since they trace a motion path in real ASL rather than a fixed hand shape, and a single-frame classifier can't reliably capture that. They're better suited to a sequence model.
- `data/` and `models/` are excluded from version control. Regenerate them with `prepare_letters_dataset.py` after unzipping the downloaded kaggle dataset. 

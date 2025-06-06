

# 🎙️ Deepfake Detector

A Python-based tool to detect deepfake audio in both video and audio files using LSTM or Xception neural network architectures. This project extracts audio from video (if necessary), preprocesses it into the appropriate format, and uses a trained model to classify it as **real** or **fake**.

## 📌 Features

* Supports both **audio** and **video** files (`.wav`, `.mp3`, `.mp4`, `.avi`, `.mov`)
* Extracts and processes audio using `librosa`, `pydub`, and `moviepy`
* Supports two model types:

  * `LSTM` for sequential audio analysis
  * `Xception` for image-based spectrogram analysis
* Batch prediction with multithreading
* Easy integration and training from datasets of real and fake samples

---

## 🧠 Model Architectures

### LSTM

* Designed for time-domain features
* Input shape: `(100, 1)` (e.g., 2-second audio padded)
* Two LSTM layers + Dense layers

### Xception

* Accepts spectrogram images resized to `(128, 128, 3)`
* Suitable for frequency-domain feature learning

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-audio-detector.git
cd deepfake-audio-detector

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

**requirements.txt**

```txt
numpy
librosa
tensorflow
pydub
moviepy
soundfile
```

> Note: `ffmpeg` must be installed and accessible in your system's PATH for audio extraction to work.
> You can install it via:

* **Ubuntu/Debian:** `sudo apt install ffmpeg`
* **Mac (Homebrew):** `brew install ffmpeg`
* **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/)

---

## 🚀 Usage

### 1. Training

```python
from deepfake_detector import DeepfakeAudioDetector

real_samples = ["real1.wav", "real2.mp4"]
fake_samples = ["fake1.wav", "fake2.mp4"]

detector = DeepfakeAudioDetector(model_type='lstm')
detector.train(real_samples, fake_samples, epochs=10, batch_size=8)
```

### 2. Predict a Single File

```python
result = detector.predict("unknown_sample.mp4")
print(result)
```

Output:

```json
{
  "file": "unknown_sample.mp4",
  "type": "audio+video",
  "label": "fake",
  "confidence": 0.92
}
```

### 3. Batch Prediction

```python
files = ["clip1.mp4", "clip2.wav"]
results = detector.batch_predict(files)

for r in results:
    print(r)
```

---

## 🧪 File Type Detection

The system automatically detects and handles:

* `audio-only` (WAV, MP3)
* `video-only` (no audio)
* `audio+video`
* `corrupted` or `unsupported` formats

---

## 📂 File Structure

```
deepfake_audio_detector/
├── deepfake_detector.py     # Main class implementation
├── gui_app.py
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```




# DeepGuard X — Multimodal Deepfake Detection

Professional Flask web app for uploading image, audio, or video media and running inference with trained Keras models.

## Features
- Responsive, professional dark forensic-analysis UI
- Image, audio, and video upload validation
- Temporary upload files are deleted after analysis
- Image model inference using `models/image_model.keras`
- Audio log-Mel preprocessing and inference using `models/audio_model.keras`
- Video frame sampling (up to 8 uniformly spaced frames) and visual score averaging
- Explicit warning when trained model weights are not present

## Requirements
Windows 10/11, Python 3.10 recommended. TensorFlow installation can be heavy; use a virtual environment.

## Run on Windows (PowerShell)
```powershell
cd D:\DeepGuard-X
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Model input/output contract
The app expects:
- `models/image_model.keras`: input `(batch, 224, 224, 3)`, sigmoid output where higher means manipulated.
- `models/audio_model.keras`: input `(batch, 128, 256, 1)`, sigmoid output where higher means manipulated.

These files are **not included** because they must be trained on a suitable dataset. Without them the website works and clearly reports that a model is required; it will not fabricate a deepfake prediction.

## Dataset and evaluation
The project plan uses FakeAVCeleb as the primary multimodal research dataset, with FaceForensics++ for additional visual examples and ASVspoof for audio-only experiments. Follow each dataset's license, access conditions, and research-use terms. Start with a small, balanced subset, split by identity/source video to reduce leakage, and report accuracy, precision, recall, F1, confusion matrix and ROC-AUC only after evaluation on held-out data.

## Current project scope / limitations
This is a runnable web/inference foundation, not a completed trained detector. Video currently samples frames and applies the image model; synchronized audio-video fusion, Whisper transcription, NLP consistency scoring, model training scripts, and benchmark metrics remain future milestones. Do not describe the project as 100% accurate.

## Complexity
- Upload validation: O(1) aside from filename handling.
- Video sampling: seeks to at most 8 frames, then processes O(k × H × W) pixels for fixed frame size; actual decoder seek cost depends on codec/container.
- Image inference: O(model inference cost), fixed 224×224 input.
- Audio preprocessing: O(N log N) approximately for STFT-based feature extraction, capped at 30 seconds and fixed 128×256 model input.
- The dominant cost is model inference and media decoding, not the Flask route itself.

## Privacy and security notes
- Uploads are deleted in a `finally` block after analysis.
- For a public deployment, add authentication, rate limits, malware scanning, HTTPS, and background jobs. Do not run Flask debug mode in production.

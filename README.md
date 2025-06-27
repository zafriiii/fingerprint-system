# Fingerprint Recognition System with AI and Cybersecurity

A secure biometric recognition prototype integrating fingerprint liveness detection using CNNs, identity matching via OpenCV, and enhanced privacy through federated learning and differential privacy. Built for final year research and presentation.

## Project Overview
This project aims to:

- Detect spoofed vs. real fingerprint images using AI (ResNet18, PyTorch)
- Match real fingerprints for identity verification
- Secure biometric data with Federated Learning (FL) and Differential Privacy (DP)
- Provide modular, real-time fingerprint recognition pipeline
- Visualize and evaluate system performance metrics in a web GUI
- Ensure robust, unified metrics logging and model handling
- Support domain generalization and sensor adaptation

---

## Features

### Liveness Detection (AI)
- CNN-based detection using a custom ResNet18 backbone (PyTorch, Opacus compatible)
- Trained on augmented fingerprint datasets with strong, identical augmentations in all scripts
- Output: Probability and decision (Live/Spoof) for each fingerprint

### Fingerprint Matching
- ORB + BFMatcher via OpenCV
- Keypoint-based comparison of fingerprint images
- Match confidence threshold configurable

### Privacy & Security
- **Federated Learning** using Flower
- **Differential Privacy** using Opacus (PyTorch)
- Simulated multi-client training with epsilon control
- Model weights and metrics updated after each federated round
- Only the best model is used for inference and further training

### Model Evaluation & Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix, Robustness Against Spoofing (TNR), Binary Cross Entropy
- All metrics visualized in the web app
- Validation split and evaluation included in both standalone and federated scripts
- CLI/utility scripts for metrics cleaning and reset

### Web Application
- Upload fingerprint and template for liveness detection and matching
- View authentication result (Live/Spoof, Match/Mismatch)
- Performance metrics tab displays metrics, confusion matrix, and bar charts
- Incident logging and user/admin alert for spoof detection

---

## File Structure

```bash
├── liveness_detection_model.py         # CNN model training, validation, metrics export (main, PyTorch)
├── liveness_detection_keras.py         # Keras model (alternative, empty by default)
├── federated_server.py                 # Flower federated server (no server-side eval)
├── federated_client_with_dp.py         # FL client with DP, auto-load/save model & metrics, validation
├── fingerprint_matcher.py              # OpenCV fingerprint comparison logic
├── augment_real.py                     # Data augmentation script (robust, for sensor adaptation)
├── app.py                              # Streamlit web GUI (main demo, metrics visualization)
├── performance_metrics.py              # Metrics calculation utilities
├── metrics.csv                         # Auto-generated performance metrics (ignored by git)
├── liveness_model_best.pth             # Trained model weights (ignored by git)
├── .gitignore                          # Ignores models, metrics, and data
├── main.py                             # CLI inference and evaluation example
├── data/                               # Dataset folder (train/val, 0_live/spoof)
```

---

## Usage: Step-by-Step

1. **Data Augmentation:**
   ```bash
   python augment_real.py
   ```
   - Augments real fingerprint images for better generalization, especially for new sensors.
2. **Model Training:**
   ```bash
   python liveness_detection_model.py
   ```
   - Trains the liveness detection model, logs metrics, and saves only the best model.
3. **Federated Learning:**
   - Start server:
     ```bash
     python federated_server.py
     ```
   - Start client(s):
     ```bash
     python federated_client_with_dp.py
     ```
   - Each client uses the same model/augmentation and logs metrics after each round.
4. **Run Web App:**
   ```bash
   streamlit run app.py
   ```
   - Upload fingerprints, view liveness/matching results, and visualize metrics.
5. **CLI Inference & Evaluation:**
   ```bash
   python main.py
   ```
   - Example for running liveness and matching from the command line.

---

## Best Practices & Troubleshooting

- **Domain Shift/Sensor Bias:**
  - If new sensor data (e.g., FX2000) is misclassified, add a small amount of its data to training or use stronger augmentation.
  - Tune the liveness threshold in the app for best TPR/FPR on new sensors.
  - Consider domain adaptation or sensor-specific preprocessing for better generalization.
- **Metrics Logging:**
  - All scripts write to `metrics.csv` in a unified format. Use the web app or CLI utilities to compare runs.
- **Model Handling:**
  - Only `liveness_model_best.pth` is used for inference and further training. Do not use intermediate checkpoints.
- **Validation Workflow:**
  - Both standalone and federated scripts use a validation split and log validation metrics after each run/round.
- **Augmentation:**
  - Use `augment_real.py` to increase data diversity and simulate sensor variability.
- **Incident Logging:**
  - Spoof detection events are logged and trigger user/admin alerts in the web app.

---

## Authors
- Developed by: Muhamad Zafri Bin Wahab
- Supervised by: Dr. Nur Erlida Binti Ruslan
- University: Multimedia University (MMU), Cyberjaya

---

## License
This project is intended for academic and educational use only.
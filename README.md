# Fingerprint Recognition System with AI and Cybersecurity

A secure biometric recognition prototype integrating fingerprint liveness detection using CNNs, identity matching via OpenCV, and enhanced privacy through federated learning and differential privacy. Built for final year research and presentation.

## Project Overview
This project aims to:

- Detect spoofed vs. real fingerprint images using AI (ResNet18)
- Match real fingerprints for identity verification
- Secure biometric data with Federated Learning (FL) and Differential Privacy (DP)
- Provide modular, real-time fingerprint recognition pipeline
- Visualize and evaluate system performance metrics in a web GUI

---

## Features

### Liveness Detection (AI)
- CNN-based detection using ResNet18 (PyTorch)
- Trained on augmented fingerprint datasets
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

### Model Evaluation & Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix, Robustness Against Spoofing, Binary Cross Entropy
- All metrics saved to `metrics.csv` and visualized in the web app

### Web Application (Streamlit)
- Upload fingerprint and template for liveness detection and matching
- View authentication result (Live/Spoof, Match/Mismatch)
- Performance metrics tab: auto-loads `metrics.csv` and displays metrics, confusion matrix, and bar charts

---

## File Structure

```bash
├── liveness_detection_model.py         # CNN model training and metrics export (main)
├── liveness_detection_keras.py         # Keras model (alternative)
├── federated_server.py                 # Flower federated server
├── federated_client_with_dp.py         # FL client with DP integration, auto-load/save model & metrics
├── fingerprint_matcher.py              # OpenCV fingerprint comparison logic
├── augment_real.py                     # Data augmentation script
├── app.py                              # Streamlit web GUI (main demo)
├── performance_metrics.py              # Metrics calculation utilities
├── metrics.csv                         # Auto-generated performance metrics
├── liveness_model.pth                  # Trained model weights
├── data/                               # Dataset folder (train/val, 0_live/spoof)
```

---

## Usage: Step-by-Step

1. **Data Augmentation:**
   ```bash
   python augment_real.py
   ```
2. **Model Training:**
   ```bash
   python liveness_detection_model.py
   ```
3. **Federated Learning:**
   - Start server:
     ```bash
     python federated_server.py
     ```
   - Start client(s):
     ```bash
     python federated_client_with_dp.py
     ```
4. **Run Web App:**
   ```bash
   streamlit run app.py
   ```

---

## Authors
- Developed by: Muhamad Zafri Bin Wahab
- Supervised by: Dr. Nur Erlida Binti Ruslan
- University: Multimedia University (MMU), Cyberjaya

---

## License
This project is intended for academic and educational use only.
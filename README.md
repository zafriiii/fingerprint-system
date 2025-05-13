
# Fingerprint Recognition System with AI and Cybersecurity

A secure biometric recognition prototype integrating fingerprint liveness detection using CNNs, identity matching via OpenCV, and enhanced privacy through federated learning and differential privacy. Built for final year research and presentation.

## Project Overview
This project aims to:

- Detect spoofed vs. real fingerprint images using AI (ResNet18)
- Match real fingerprints for identity verification
- Secure biometric data with Federated Learning (FL) and Differential Privacy (DP)
- Provide modular, real-time fingerprint recognition pipeline

---

## Features

### Liveness Detection (AI)
- CNN-based detection using ResNet18
- Trained on fingerprint datasets (`live` vs `spoof` folders)
- Output: Probability of being a live fingerprint

### Fingerprint Matching
- ORB + BFMatcher via OpenCV
- Keypoint-based comparison of fingerprint images
- Match confidence threshold configurable

### Privacy & Security
- **Federated Learning** using Flower
- **Differential Privacy** using Opacus (PyTorch)
- Simulated multi-client training with epsilon control

### Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- False Acceptance Rate (FAR), False Rejection Rate (FRR)

---

## File Structure

```bash
├── liveness_detection_model.py         # CNN model training using PyTorch
├── liveness_detection_keras.py         # Optional: Keras model (alternative version)
├── differential_privacy.py             # Training with DP via Opacus
├── federated_server.py                 # Flower federated server
├── federated_client.py                 # FL client (basic)
├── federated_client_with_dp.py         # FL client with DP integration
├── fingerprint_matcher.py              # OpenCV fingerprint comparison logic
├── main.py                             # Unified entry point for testing liveness + matching
├── setup.bat                           # Batch script to run virtual environment and scripts
├── venv/                               # Python virtual environment (excluded from versioning)
└── .vscode/                            # VS Code workspace settings (optional)
```

---

## Getting Started

1. Clone the repository
2. Activate your virtual environment:
   ```bash
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train model:
   ```bash
   python liveness_detection_model.py
   ```
5. Run prototype pipeline:
   ```bash
   python main.py
   ```

---

## Demonstration Workflow

1. Load fingerprint image using OpenCV
2. Run liveness detection (ResNet18)
3. If result is **Live**:
   - Match fingerprint with known template
   - Output result: `Match` or `Mismatch`
4. If **Spoofed**: Reject immediately

---

## Authors
- Developed by: [Your Name]
- Supervised by: [Supervisor Name]
- University: [University Name]

---

## License
This project is intended for academic and educational use only.

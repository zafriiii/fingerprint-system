import cv2
import torch
import torchvision.transforms as transforms
from liveness_detection_model import FingerprintLivenessCNN
from fingerprint_matcher import match_fingerprints
from performance_metrics import calculate_metrics, get_confusion_matrix, robustness_against_spoofing, binary_cross_entropy

IMAGE_SIZE = 224

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found.")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    return tensor

def check_liveness(model, tensor):
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        prob = output.item()
        print(f"Liveness score: {prob:.4f}")
        return "Live" if prob > 0.45 else "Spoof"

def evaluate_performance(y_true, y_pred, y_prob=None):
    metrics = calculate_metrics(y_true, y_pred)
    print("Performance Metrics:", metrics)
    print("Confusion Matrix:\n", get_confusion_matrix(y_true, y_pred))
    print("Robustness against spoofing:", robustness_against_spoofing(y_true, y_pred))
    if y_prob is not None:
        print("Binary Cross Entropy:", binary_cross_entropy(y_true, y_prob))

if __name__ == "__main__":
    live_model = FingerprintLivenessCNN()
    live_model.load_state_dict(torch.load("liveness_model.pth", map_location=torch.device('cpu')))

    fingerprint_path = "test_input.png"
    enrolled_path = "enrolled_template.png"

    try:
        tensor = preprocess_image(fingerprint_path)
        result = check_liveness(live_model, tensor)

        if result == "Live":
            print("Fingerprint is LIVE. Proceeding to match...")
            match_result = match_fingerprints(fingerprint_path, enrolled_path)
            print(match_result)
        else:
            print("Fingerprint is SPOOFED. Access denied.")

        # Example: Evaluate performance (replace with your real data)
        # y_true = [0, 1, 0, 1]  # 0=live, 1=spoof
        # y_pred = [0, 1, 1, 0]
        # y_prob = [0.9, 0.8, 0.4, 0.3]  # predicted probabilities
        # evaluate_performance(y_true, y_pred, y_prob)

    except Exception as e:
        print("Error:", str(e))

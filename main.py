
import cv2
import torch
import torchvision.transforms as transforms
from liveness_detection_model import FingerprintLivenessCNN
from fingerprint_matcher import match_fingerprints

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
        return "Live" if prob > 0.5 else "Spoof"

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
    except Exception as e:
        print("Error:", str(e))

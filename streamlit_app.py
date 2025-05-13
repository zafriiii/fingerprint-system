
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from liveness_detection_model import FingerprintLivenessCNN
from fingerprint_matcher import match_fingerprints

# Constants
IMAGE_SIZE = 224
MODEL_PATH = "liveness_model.pth"
ENROLLED_IMAGE = "enrolled_template.png"  # Replace with path to enrolled fingerprint

# Load model
@st.cache_resource
def load_model():
    model = FingerprintLivenessCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess image for model
def preprocess_image(image):
    img = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    return tensor

# Check liveness
def check_liveness(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        score = output.item()
    return score, "Live" if score > 0.5 else "Spoof"

# App UI
st.title("Fingerprint Recognition System")
st.write("Upload a fingerprint image to check for liveness and match.")

uploaded_file = st.file_uploader("Choose a fingerprint image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fingerprint", use_column_width=True)

    tensor = preprocess_image(image)
    model = load_model()
    
    score, result = check_liveness(model, tensor)
    st.markdown(f"### Liveness Result: `{result}` (Score: {score:.4f})")

    if result == "Live":
        st.markdown("### Proceeding to fingerprint matching...")
        cv2.imwrite("temp_input.png", np.array(image.convert("L")))
        match_result = match_fingerprints("temp_input.png", ENROLLED_IMAGE)
        st.success(match_result)
    else:
        st.error("Spoofed fingerprint. Matching skipped.")

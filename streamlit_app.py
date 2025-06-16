
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# Configuration
MODEL_PATH = 'liveness_model.pth'
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
class FingerprintLivenessCNN(nn.Module):
    def __init__(self):
        super(FingerprintLivenessCNN, self).__init__()
        from torchvision import models
        self.base_model = models.resnet18(pretrained=False)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

@st.cache_resource
def load_model():
    model = FingerprintLivenessCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Streamlit UI 
st.title("Fingerprint Liveness Detection")
st.write("Upload a fingerprint image to check if it's **Live** or **Spoofed**.")

uploaded_file = st.file_uploader("Choose a fingerprint image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Fingerprint', use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor).item()

    st.write("### Prediction:")
    if output < 0.5:
        confidence = 1 - output  # confidence for Live
        st.success(f"Live Fingerprint (Confidence: {confidence:.4f})")
    else:
        confidence = output  # confidence for Spoof
        st.error(f"Spoofed Fingerprint (Confidence: {confidence:.4f})")

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from fingerprint_matcher import match_fingerprints
from performance_metrics import calculate_metrics, get_confusion_matrix, robustness_against_spoofing, binary_cross_entropy

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
st.title("Fingerprint Liveness Detection and Matching")
tab1, tab2 = st.tabs(["Liveness & Matching", "Performance Metrics"])

with tab1:
    st.write("Upload two fingerprint images: one for liveness detection, one enrolled template for matching.")
    uploaded_file = st.file_uploader("Fingerprint for Liveness Detection", type=["jpg", "jpeg", "png", "tif"], key="live")
    enrolled_file = st.file_uploader("Enrolled Template for Matching", type=["jpg", "jpeg", "png", "tif"], key="enroll")

    if uploaded_file is not None and enrolled_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Fingerprint', use_container_width=True)
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor).item()
        st.write("### Liveness Prediction:")
        if output < 0.5:
            confidence = 1 - output
            st.success(f"Live Fingerprint (Confidence: {confidence:.4f})")
            # Save uploaded files temporarily
            live_path = "temp_live.png"
            enroll_path = "temp_enroll.png"
            image.save(live_path)
            Image.open(enrolled_file).convert("RGB").save(enroll_path)
            # Run matcher
            match_result = match_fingerprints(live_path, enroll_path)
            st.write(f"### Matcher Result: {match_result}")
            # Clean up
            os.remove(live_path)
            os.remove(enroll_path)
        else:
            confidence = output
            st.error(f"Spoofed Fingerprint (Confidence: {confidence:.4f})")

with tab2:
    st.write("### Performance Metrics")
    st.write("Upload your ground truth and prediction files (CSV) or use example data.")
    uploaded_metrics = st.file_uploader("Upload CSV with columns: y_true, y_pred, y_prob (optional)", type=["csv"], key="metrics")
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    if uploaded_metrics is not None:
        df = pd.read_csv(uploaded_metrics)
        y_true = df['y_true']
        y_pred = df['y_pred']
        y_prob = df['y_prob'] if 'y_prob' in df.columns else None
    elif os.path.exists('metrics.csv'):
        df = pd.read_csv('metrics.csv')
        st.info("Loaded metrics.csv automatically.")
        y_true = df['y_true']
        y_pred = df['y_pred']
        y_prob = df['y_prob'] if 'y_prob' in df.columns else None
    else:
        y_true = [0, 1, 0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 1, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.8, 0.4, 0.3, 0.7, 0.2, 0.85, 0.1]
    metrics = calculate_metrics(y_true, y_pred)
    st.write("**Metrics:**", metrics)
    cm = get_confusion_matrix(y_true, y_pred)
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)
    st.write("**Robustness against spoofing:**", robustness_against_spoofing(y_true, y_pred))
    if y_prob is not None:
        st.write("**Binary Cross Entropy:**", binary_cross_entropy(y_true, y_prob))
    # Show bar chart for metrics
    st.write("**Metrics Bar Chart:**")
    fig2, ax2 = plt.subplots()
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    ax2.bar(metric_names, metric_values, color='skyblue')
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)

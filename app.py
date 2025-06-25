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
    st.write("Upload a fingerprint image for liveness detection.")
    liveness_threshold = st.slider("Liveness threshold (lower = stricter, higher = more tolerant)", min_value=0.3, max_value=0.7, value=0.5, step=0.01)
    uploaded_file = st.file_uploader("Fingerprint for Liveness Detection", type=["jpg", "jpeg", "png", "tif"], key="live")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Fingerprint', use_container_width=True)
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor).item()
        st.write("### Liveness Prediction:")
        if output < liveness_threshold:
            confidence = 1 - output
            st.success(f"Live Fingerprint (Confidence: {confidence:.4f})")
            # Show enrolled template uploader only if live
            enrolled_file = st.file_uploader("Enrolled Template for Matching", type=["jpg", "jpeg", "png", "tif"], key="enroll")
            if enrolled_file is not None:
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
            st.warning("Matcher is disabled for spoofed fingerprints.")

with tab2:
    st.write("### Performance Metrics")
    st.write("Upload your ground truth and prediction files (CSV) or use example data.")
    uploaded_metrics = st.file_uploader("Upload CSV with columns: run_id, epoch, timestamp, y_true, y_pred, y_prob (optional)", type=["csv"], key="metrics")
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    if uploaded_metrics is not None:
        df = pd.read_csv(uploaded_metrics)
        st.write("**Raw Metrics Data:**")
        st.dataframe(df)
    elif os.path.exists('metrics.csv'):
        df = pd.read_csv('metrics.csv')
        st.info("Loaded metrics.csv automatically.")
        st.write("**Raw Metrics Data:**")
        st.dataframe(df)
    else:
        df = pd.DataFrame({
            'run_id': ['example']*8,
            'epoch': [20]*8,
            'timestamp': ['example']*8,
            'y_true': [0, 1, 0, 1, 1, 0, 1, 0],
            'y_pred': [0, 1, 1, 0, 1, 0, 1, 0],
            'y_prob': [0.9, 0.8, 0.4, 0.3, 0.7, 0.2, 0.85, 0.1]
        })
        st.dataframe(df)

    # Option to filter by run_id if present
    if 'run_id' in df.columns:
        run_ids = df['run_id'].unique()
        selected_run = st.selectbox("Select run_id to view metrics for a specific run:", run_ids)
        filtered_df = df[df['run_id'] == selected_run]
    else:
        st.info("No run_id column found. Showing all data.")
        filtered_df = df
    y_true = filtered_df['y_true']
    y_pred = filtered_df['y_pred']
    y_prob = filtered_df['y_prob'] if 'y_prob' in filtered_df.columns else None

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    st.write("\n---\n")
    st.subheader("Summary Metrics Table")
    st.table({k: [v] for k, v in metrics.items()})

    cm = get_confusion_matrix(y_true, y_pred)
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

    robustness = robustness_against_spoofing(y_true, y_pred)
    st.write(f"**Robustness against spoofing:** {robustness}")
    if y_prob is not None:
        bce = binary_cross_entropy(y_true, y_prob)
        st.write(f"**Binary Cross Entropy:** {bce}")

    # Show bar chart for metrics
    st.write("**Metrics Bar Chart:**")
    fig2, ax2 = plt.subplots()
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    ax2.bar(metric_names, metric_values, color='skyblue')
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)

    # Calculate metrics for each run_id for comparison chart
    if 'run_id' in df.columns and len(df['run_id'].unique()) > 1:
        st.write("\n---\n")
        st.subheader("Comparison Across Runs")
        run_metrics = {}
        for rid in df['run_id'].unique():
            sub = df[df['run_id'] == rid]
            y_true_sub = sub['y_true']
            y_pred_sub = sub['y_pred']
            run_metrics[rid] = calculate_metrics(y_true_sub, y_pred_sub)
        import numpy as np
        metric_names = list(next(iter(run_metrics.values())).keys())
        x = np.arange(len(metric_names))
        width = 0.8 / len(run_metrics)  # width of each bar
        fig3, ax3 = plt.subplots()
        for i, (rid, metrics_dict) in enumerate(run_metrics.items()):
            values = [metrics_dict[m] for m in metric_names]
            ax3.bar(x + i*width, values, width, label=str(rid))
        ax3.set_xticks(x + width*(len(run_metrics)-1)/2)
        ax3.set_xticklabels(metric_names)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Score')
        ax3.set_title('Metrics Comparison Across Runs')
        ax3.legend(title='run_id')
        st.pyplot(fig3)

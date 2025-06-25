import os

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from fingerprint_matcher import match_fingerprints
from performance_metrics import (binary_cross_entropy, calculate_metrics,
                                 get_confusion_matrix,
                                 robustness_against_spoofing)

# Configuration
MODEL_PATH = "liveness_model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            nn.Sigmoid(),
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
transform = transforms.Compose(
    [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
)

# Streamlit UI
st.title("Fingerprint Liveness Detection and Matching")
tab1, tab2 = st.tabs(["Liveness & Matching", "Performance Metrics"])

with tab1:
    st.write("Upload a fingerprint image for liveness detection.")
    liveness_threshold = st.slider(
        "Liveness threshold (lower = stricter, higher = more tolerant)",
        min_value=0.3,
        max_value=0.7,
        value=0.5,
        step=0.01,
    )
    uploaded_file = st.file_uploader(
        "Fingerprint for Liveness Detection",
        type=["jpg", "jpeg", "png", "tif"],
        key="live",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Fingerprint", use_container_width=True)
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor).item()
        st.write("### Liveness Prediction:")
        if output < liveness_threshold:
            confidence = 1 - output
            st.success(f"Live Fingerprint (Confidence: {confidence:.4f})")
            # Show enrolled template uploader only if live
            enrolled_file = st.file_uploader(
                "Enrolled Template for Matching",
                type=["jpg", "jpeg", "png", "tif"],
                key="enroll",
            )
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
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np

    # Always try to load metrics.csv
    if os.path.exists("metrics.csv"):
        df = pd.read_csv("metrics.csv")
        st.info("Loaded metrics.csv automatically.")
        st.write("**Raw Metrics Data:**")
        st.dataframe(df)
    else:
        st.warning("metrics.csv not found. Please run training or federated learning first.")
        st.stop()

    # Filter for summary rows
    if "epoch" in df.columns:
        summary_df = df[df["epoch"] == "summary"]
        if summary_df.empty:
            st.warning("No summary metrics found in metrics.csv.")
            st.stop()
    else:
        st.warning("No 'epoch' column found. Please update your metrics.csv format.")
        st.stop()

    # Option to select run_id
    run_ids = summary_df["run_id"].unique()
    selected_run = st.selectbox("Select run_id to view metrics for a specific run:", run_ids)
    run_metrics = summary_df[summary_df["run_id"] == selected_run].iloc[0]

    # Show summary metrics table
    metrics_table = {
        "Accuracy": run_metrics.get("accuracy", None),
        "Precision": run_metrics.get("precision", None),
        "Recall": run_metrics.get("recall", None),
        "F1-Score": run_metrics.get("f1_score", None),
        "Robustness (TNR)": run_metrics.get("robustness_tnr", None),
        "Binary Cross Entropy": run_metrics.get("bce", None),
    }
    st.subheader("Summary Metrics Table")
    st.table({k: [v] for k, v in metrics_table.items()})

    # Bar chart for metrics
    st.write("**Metrics Bar Chart:**")
    metric_names = [k for k in metrics_table.keys() if k != "Binary Cross Entropy"]
    metric_values = [float(run_metrics.get(col.lower().replace(" ", "_"), 0)) for col in metric_names]
    fig2, ax2 = plt.subplots()
    ax2.bar(metric_names, metric_values, color="skyblue")
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)

    # Show BCE separately
    st.write(f"**Binary Cross Entropy:** {run_metrics.get('bce', None)}")

    # Comparison across runs
    if len(run_ids) > 1:
        st.write("\n---\n")
        st.subheader("Comparison Across Runs")
        compare_metrics = ["accuracy", "precision", "recall", "f1_score", "robustness_tnr"]
        x = np.arange(len(compare_metrics))
        width = 0.8 / len(run_ids)
        fig3, ax3 = plt.subplots()
        for i, rid in enumerate(run_ids):
            row = summary_df[summary_df["run_id"] == rid].iloc[0]
            values = [float(row.get(m, 0)) for m in compare_metrics]
            ax3.bar(x + i * width, values, width, label=str(rid))
        ax3.set_xticks(x + width * (len(run_ids) - 1) / 2)
        ax3.set_xticklabels([m.replace("_", " ").title() for m in compare_metrics])
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Score")
        ax3.set_title("Metrics Comparison Across Runs")
        ax3.legend(title="run_id")
        st.pyplot(fig3)

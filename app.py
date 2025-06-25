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

    # Helper to safely convert to float for metrics
    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # Show each metric individually
    st.subheader("Summary Metrics")
    acc = safe_float(run_metrics.get('accuracy', 0))
    prec = safe_float(run_metrics.get('precision', 0))
    rec = safe_float(run_metrics.get('recall', 0))
    f1 = safe_float(run_metrics.get('f1_score', 0))
    tnr = safe_float(run_metrics.get('robustness_tnr', 0))
    bce = safe_float(run_metrics.get('bce', 0))
    st.metric("Accuracy", f"{acc:.4f}" if acc is not None else "N/A")
    st.metric("Precision", f"{prec:.4f}" if prec is not None else "N/A")
    st.metric("Recall", f"{rec:.4f}" if rec is not None else "N/A")
    st.metric("F1-Score", f"{f1:.4f}" if f1 is not None else "N/A")
    st.metric("Robustness (TNR)", f"{tnr:.4f}" if tnr is not None else "N/A")
    st.metric("Binary Cross Entropy", f"{bce:.4f}" if bce is not None else "N/A")

    # Calculate and display processing time per sample (ms)
    st.subheader("Processing Time")
    if "timestamp" in df.columns and "run_id" in df.columns:
        per_sample_rows = df[(df["run_id"] == selected_run) & (df["epoch"] != "summary")]
        if not per_sample_rows.empty:
            if "processing_time" in per_sample_rows.columns:
                total_time_ms = per_sample_rows["processing_time"].sum()
                num_samples = len(per_sample_rows)
                avg_time_ms = total_time_ms / num_samples if num_samples > 0 else None
            else:
                avg_time_ms = None
            st.metric("Processing Time per Sample (ms)", f"{avg_time_ms:.2f}" if avg_time_ms is not None else "N/A")
        else:
            st.write("**Processing Time per Sample:** N/A (no per-sample data)")
    else:
        st.write("**Processing Time per Sample:** N/A (missing columns)")

    # Bar chart for metrics
    st.subheader("Metrics Bar Chart")
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Robustness (TNR)"]
    metric_vals = [acc, prec, rec, f1, tnr]
    metric_plot_vals = [v if v is not None else 0 for v in metric_vals]
    fig2, ax2 = plt.subplots()
    ax2.bar(metric_names, metric_plot_vals, color="skyblue")
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)

    # Bar chart for Binary Cross Entropy
    st.subheader("Binary Cross Entropy Bar Chart")
    fig_bce, ax_bce = plt.subplots()
    ax_bce.bar(["Binary Cross Entropy"], [bce if bce is not None else 0], color="salmon")
    ax_bce.set_ylabel("BCE")
    st.pyplot(fig_bce)

    # Line chart for Binary Cross Entropy over epochs (if available)
    st.subheader("Binary Cross Entropy Over Epochs")
    # Try to get per-epoch BCE for the selected run
    if "epoch" in df.columns and "bce" in df.columns:
        bce_epochs = df[(df["run_id"] == selected_run) & (df["epoch"] != "summary") & (df["bce"] != '')]
        if not bce_epochs.empty:
            # Convert epoch and bce to numeric, drop rows with invalid values
            bce_epochs = bce_epochs.copy()
            bce_epochs["epoch_num"] = pd.to_numeric(bce_epochs["epoch"], errors="coerce")
            bce_epochs["bce_val"] = pd.to_numeric(bce_epochs["bce"], errors="coerce")
            bce_epochs = bce_epochs.dropna(subset=["epoch_num", "bce_val"])
            if not bce_epochs.empty:
                fig_bce_line, ax_bce_line = plt.subplots()
                ax_bce_line.plot(bce_epochs["epoch_num"], bce_epochs["bce_val"], marker='o', color='salmon')
                ax_bce_line.set_xlabel("Epoch")
                ax_bce_line.set_ylabel("Binary Cross Entropy")
                ax_bce_line.set_title("BCE Over Epochs")
                st.pyplot(fig_bce_line)
            else:
                st.info("No valid per-epoch BCE data available for this run.")
        else:
            st.info("No per-epoch BCE data available for this run.")
    else:
        st.info("No per-epoch BCE data available in metrics.csv.")

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

    # Confusion Matrix Heatmap
    st.subheader("Confusion Matrix")
    if "conf_matrix" in run_metrics:
        cm_str = run_metrics["conf_matrix"]
        try:
            cm_vals = [int(x) for x in cm_str.split(",") if x.strip() != ""]
            if len(cm_vals) == 4:
                cm = np.array(cm_vals).reshape(2, 2)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                ax_cm.set_title("Confusion Matrix")
                st.pyplot(fig_cm)
            else:
                st.info("Confusion matrix data is incomplete for this run.")
        except Exception as e:
            st.info(f"Could not parse confusion matrix: {e}")
    else:
        st.info("No confusion matrix data available in metrics.csv.")

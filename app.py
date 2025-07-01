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

MODEL_PATH = "liveness_model_best.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import models

class PatchedBasicBlock(BasicBlock):
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

def patched_resnet18():
    return ResNet(block=PatchedBasicBlock, layers=[2, 2, 2, 2])

class FingerprintLivenessCNN(nn.Module):
    def __init__(self):
        super(FingerprintLivenessCNN, self).__init__()
        self.resnet = patched_resnet18()
        try:
            state_dict = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).state_dict()
            self.resnet.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
        for name, param in self.resnet.named_parameters():
            if "layer4" in name or "layer3" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self._set_relu_inplace(self.resnet)
        self._set_relu_inplace(self.classifier)

    def _set_relu_inplace(self, module):
        for m in module.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return torch.sigmoid(x)

@st.cache_resource
def load_model():
    model = FingerprintLivenessCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose(
    [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
)

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
            enrolled_file = st.file_uploader(
                "Enrolled Template for Matching",
                type=["jpg", "jpeg", "png", "tif"],
                key="enroll",
            )
            if enrolled_file is not None:
                live_path = "temp_live.png"
                enroll_path = "temp_enroll.png"
                image.save(live_path)
                Image.open(enrolled_file).convert("RGB").save(enroll_path)
                match_result = match_fingerprints(live_path, enroll_path)
                st.write(f"### Matcher Result: {match_result}")
                os.remove(live_path)
                os.remove(enroll_path)
        else:
            confidence = output
            st.error(f"Spoofed Fingerprint (Confidence: {confidence:.4f})")
            st.warning("Matcher is disabled for spoofed fingerprints.")

            from datetime import datetime
            with open("spoof_incidents.log", "a") as log_file:
                log_file.write(f"{datetime.now().isoformat()} - Spoof detected! Confidence: {confidence:.4f}\n")

            st.warning("ALERT: Spoofed fingerprint detected! Incident has been logged and the administrator has been notified.")

with tab2:
    st.write("### Performance Metrics")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px

    if os.path.exists("metrics.csv"):
        df = pd.read_csv("metrics.csv")
        st.info("metrics.csv successfully loaded")
        st.write("**Raw Metrics Data:**")
        st.dataframe(df)
    else:
        st.warning("metrics.csv not found. Please run training or federated learning first.")
        st.stop()

    summary_df = df[df["epoch"].isin(["summary", "val_summary"])]
    if summary_df.empty:
        st.warning("No summary metrics found in metrics.csv.")
        st.stop()

    summary_df["label"] = summary_df.apply(
        lambda row: f"{row['run_id']} ({row['source']})", axis=1
    )
    selected_label = st.selectbox(
        "Select run to view metrics for a specific run and method:",
        summary_df["label"].values,
    )
    run_metrics = summary_df[summary_df["label"] == selected_label].iloc[0]
    st.info(f"**Training Source:** {run_metrics.get('source', 'Unknown')}")
    selected_run = run_metrics["run_id"]
    run_ids = summary_df["run_id"].unique()
    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    st.subheader("Summary Metrics", help="Accuracy: Fraction of correct predictions. Precision: Of all predicted live, how many are truly live. Recall: Of all actual live, how many detected. F1-Score: Harmonic mean of precision and recall. Robustness (TNR): True Negative Rate. BCE: Average loss per sample.")
    acc = safe_float(run_metrics.get('accuracy', 0))
    prec = safe_float(run_metrics.get('precision', 0))
    rec = safe_float(run_metrics.get('recall', 0))
    f1 = safe_float(run_metrics.get('f1_score', 0))
    tnr = safe_float(run_metrics.get('robustness_tnr', 0))
    bce = safe_float(run_metrics.get('bce', 0))
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Robustness (TNR)"]
    metric_vals = [acc, prec, rec, f1, tnr]
    metric_plot_vals = [v if v is not None else 0 for v in metric_vals]
    st.metric("Accuracy", f"{acc:.4f}" if acc is not None else "N/A", help="How many predictions were correct out of all? Shows if the model is right most of the time.")
    st.metric("Precision", f"{prec:.4f}" if prec is not None else "N/A", help="Of all the times the model said 'live', how many were actually live? High precision means few false alarms.")
    st.metric("Recall", f"{rec:.4f}" if rec is not None else "N/A", help="Of all real live fingerprints, how many did the model find? High recall means it misses few real ones.")
    st.metric("F1-Score", f"{f1:.4f}" if f1 is not None else "N/A", help="A balance between precision and recall. High F1 means both are good.")
    st.metric("Robustness (TNR)", f"{tnr:.4f}" if tnr is not None else "N/A", help="How well the model says 'no' to fake or non-fingerprint images. High TNR means it rarely gets fooled.")
    st.metric("Binary Cross Entropy", f"{bce:.4f}" if bce is not None else "N/A", help="Average error per sample. Lower is better. Shows how well the model is learning.")

    st.subheader("Processing Time", help="Average time (in ms) to process one validation image on your hardware.")
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

    st.subheader("Metrics Bar Chart", help="Compares the main metrics (Accuracy, Precision, Recall, F1-Score, Robustness) for the selected run. Higher bars indicate better performance.")
    fig2 = go.Figure([go.Bar(x=metric_names, y=metric_plot_vals, marker_color="skyblue")])
    fig2.update_layout(yaxis=dict(range=[0, 1]), xaxis_title="Metric", yaxis_title="Score")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Binary Cross Entropy Over Epochs", help="Shows how the average loss (BCE) changes over training epochs. A downward trend means the model is learning.")
    if "epoch" in df.columns and "bce" in df.columns:
        bce_epochs = df[(df["run_id"] == selected_run) & (df["epoch"] != "summary") & (df["bce"] != '')]
        if not bce_epochs.empty:
            bce_epochs = bce_epochs.copy()
            bce_epochs["epoch_num"] = pd.to_numeric(bce_epochs["epoch"], errors="coerce")
            bce_epochs["bce_val"] = pd.to_numeric(bce_epochs["bce"], errors="coerce")
            bce_epochs = bce_epochs.dropna(subset=["epoch_num", "bce_val"])
            if not bce_epochs.empty:
                fig_bce_line = go.Figure([go.Scatter(x=bce_epochs["epoch_num"], y=bce_epochs["bce_val"], mode='lines+markers', marker_color='salmon')])
                fig_bce_line.update_layout(xaxis_title="Epoch", yaxis_title="Binary Cross Entropy", title="BCE Over Epochs")
                st.plotly_chart(fig_bce_line, use_container_width=True)
            else:
                st.info("No valid per-epoch BCE data available for this run.")
        else:
            st.info("No per-epoch BCE data available for this run.")
    else:
        st.info("No per-epoch BCE data available in metrics.csv.")

    if len(run_ids) > 1:
        st.write("\n---\n")
        st.subheader("Comparison Across Runs", help="Compares metrics for multiple training runs. Useful for comparing different model versions or hyperparameters.")
        compare_metrics = ["accuracy", "precision", "recall", "f1_score", "robustness_tnr"]
        x = np.arange(len(compare_metrics))
        width = 0.8 / len(run_ids)
        fig3 = go.Figure()
        for i, rid in enumerate(run_ids):
            row = summary_df[summary_df["run_id"] == rid].iloc[0]
            values = []
            for m in compare_metrics:
                val = row.get(m, 0)
                try:
                    if isinstance(val, str) and ',' in val:
                        values.append(0.0)
                    else:
                        values.append(float(val))
                except (ValueError, TypeError):
                    values.append(0.0)
            fig3.add_bar(x=[m.replace("_", " ").title() for m in compare_metrics], y=values, name=str(rid))
        fig3.update_layout(barmode='group', yaxis=dict(range=[0, 1]), yaxis_title="Score", xaxis_title="Metric", title="Metrics Comparison Across Runs")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Confusion Matrix", help="Rows: Actual class (0 = Spoof, 1 = Live). Columns: Predicted class (0 = Spoof, 1 = Live). Diagonal values are correct predictions; off-diagonal are errors.")
    cm_str = run_metrics.get("conf_matrix", "")
    try:
        if isinstance(cm_str, str):
            cm_vals = [int(x) for x in cm_str.split(",") if x.strip() != ""]
        else:
            cm_vals = []
        if len(cm_vals) == 4:
            cm = np.array(cm_vals).reshape(2, 2)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
            fig_cm.update_xaxes(title_text="Predicted")
            fig_cm.update_yaxes(title_text="Actual")
            fig_cm.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Confusion matrix data is incomplete for this run.")
    except Exception as e:
        st.info(f"Could not parse confusion matrix: {e}")

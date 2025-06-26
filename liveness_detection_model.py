import os
import uuid
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3  # Increased from 1e-4 to 1e-3 for faster convergence
IMAGE_SIZE = 224

# --- Model and augmentation must match federated_client_with_dp.py for compatibility ---
from torchvision.models.resnet import ResNet, BasicBlock

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
        # Load weights from torchvision's resnet18 if available
        try:
            from torchvision import models
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


# 2. Dataset and transforms
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Dataset path
dataset_path = "data"

train_dataset = ImageFolder(
    root=os.path.join(dataset_path, "train"), transform=train_transform
)
val_dataset = ImageFolder(
    root=os.path.join(dataset_path, "val"), transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

if __name__ == "__main__":
    # 3. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerprintLivenessCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Optionally load best model if exists (for resume or inference)
    if os.path.exists("liveness_model_best.pth"):
        model.load_state_dict(torch.load("liveness_model_best.pth", map_location=device))
        print("Loaded best model weights from liveness_model_best.pth")

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # Decrease LR by 30% every 10 epochs

    # Initialize run_id and timestamp ONCE at the top for the whole run
    run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Define a strict header and column order for all metrics.csv writes
    METRICS_HEADER = [
        "run_id", "epoch", "timestamp", "y_true", "y_pred", "y_prob", "processing_time",
        "accuracy", "precision", "recall", "f1_score", "robustness_tnr", "bce", "conf_matrix", "source"
    ]

    # 4. Training loop with early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 20  # Stop if no improvement for 20 epochs

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_count}/{len(train_loader)}] Loss: {loss.item():.4f}"
            )
        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}"
        )
        # Log per-epoch BCE to metrics.csv
        epoch_row = {
            "run_id": run_id,
            "epoch": epoch + 1,
            "timestamp": timestamp,
            "y_true": '',
            "y_pred": '',
            "y_prob": '',
            "processing_time": '',
            "accuracy": '',
            "precision": '',
            "recall": '',
            "f1_score": '',
            "robustness_tnr": '',
            "bce": avg_loss,
            "source": 'Standalone',
        }
        csv_path = "metrics.csv"
        # When writing per-epoch, per-sample, or summary rows, always use METRICS_HEADER and fill missing columns with ''
        def write_metrics_row(row, csv_path):
            import pandas as pd
            # Ensure all columns are present and in correct order
            row_filled = {k: row.get(k, '') for k in METRICS_HEADER}
            df = pd.DataFrame([row_filled])
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                df.to_csv(csv_path, header=METRICS_HEADER, index=False)
        write_metrics_row(epoch_row, csv_path)
        scheduler.step()

        # Early stopping: evaluate on validation set
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(1, val_batches)
        print(f"Validation BCE: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            # Always save best model only
            torch.save(model.state_dict(), "liveness_model_best.pth")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epochs.")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # 5. Evaluate and save metrics
    all_labels = []
    all_preds = []
    all_probs = []
    all_epochs = []
    processing_times = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().to(device)
            import time

            start = time.time()
            outputs = model(images).squeeze()
            end = time.time()
            elapsed_ms = (end - start) * 1000  # ms for the batch
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_epochs.extend([NUM_EPOCHS] * len(labels))
            # Record per-sample processing time (divide batch time by batch size)
            processing_times.extend([elapsed_ms / len(labels)] * len(labels))

    # Prepare DataFrame with extra columns (including summary metrics for each row)
    # Calculate summary metrics before creating the DataFrame
    accuracy = float(accuracy_score(all_labels, all_preds))
    precision = float(precision_score(all_labels, all_preds, zero_division=0))
    recall = float(recall_score(all_labels, all_preds, zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, zero_division=0))
    # After calculating all_preds and all_labels, add confusion matrix calculation
    cm = confusion_matrix(all_labels, all_preds)
    cm_flat = cm.flatten() if cm.size == 4 else [0, 0, 0, 0]  # [TN, FP, FN, TP]

    metrics_df = pd.DataFrame(
        {
            "run_id": [run_id] * len(all_labels),
            "epoch": all_epochs,
            "timestamp": [timestamp] * len(all_labels),
            "y_true": all_labels,
            "y_pred": all_preds,
            "y_prob": all_probs,
            "processing_time": processing_times,
            "accuracy": [accuracy] * len(all_labels),
            "precision": [precision] * len(all_labels),
            "recall": [recall] * len(all_labels),
            "f1_score": [f1] * len(all_labels),
            "conf_matrix": [','.join(map(str, cm_flat))] * len(all_labels),
            "source": ['Standalone'] * len(all_labels),
        }
    )

    # Append to CSV if exists, else create new
    csv_path = "metrics.csv"
    if os.path.exists(csv_path):
        metrics_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(csv_path, header=METRICS_HEADER, index=False)
    print("Validation metrics appended to metrics.csv")

    # 6. Save model
    # Only save best model (already saved during training if improved)
    print("Best model saved as liveness_model_best.pth")

    # Calculate summary metrics
    accuracy = float(accuracy_score(all_labels, all_preds))
    precision = float(precision_score(all_labels, all_preds, zero_division=0))
    recall = float(recall_score(all_labels, all_preds, zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, zero_division=0))
    # Robustness against spoofing (TNR)
    import numpy as np
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    tn = ((all_labels_np == 0) & (all_preds_np == 0)).sum()
    fp = ((all_labels_np == 0) & (all_preds_np == 1)).sum()
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    # Binary Cross Entropy (BCE)
    if len(all_probs) > 0 and len(all_labels) > 0:
        bce = float(nn.BCELoss()(torch.tensor(all_probs), torch.tensor(all_labels)).item())
        if np.isnan(bce) or np.isinf(bce):
            bce = 0.0
    else:
        bce = 0.0
    # In summary row, add conf_matrix (always 4 values, even if all zero)
    if not (isinstance(cm_flat, (list, np.ndarray)) and len(cm_flat) == 4):
        cm_flat = [0, 0, 0, 0]
    # When writing the summary row, always use METRICS_HEADER order
    summary_row = {
        'run_id': run_id,
        'epoch': 'summary',
        'timestamp': timestamp,
        'y_true': '',
        'y_pred': '',
        'y_prob': '',
        'processing_time': '',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'robustness_tnr': tnr,
        'bce': bce,
        'conf_matrix': ','.join(map(str, cm_flat)),
        'source': 'Standalone',
    }
    write_metrics_row(summary_row, csv_path)
    print("Summary metrics appended to metrics.csv")

    # 7. Fingerrint Prediction
    def predict_single_image(image_path):
        import cv2

        model.eval()
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(tensor)
            result = "Live" if output.item() > 0.5 else "Spoof"
        return result

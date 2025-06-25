import os
import uuid
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224


# 1. CNN model using ResNet18
class FingerprintLivenessCNN(nn.Module):
    def __init__(self):
        super(FingerprintLivenessCNN, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.base_model(x)


# 2. Dataset and transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

# Dataset path
dataset_path = "data"

train_dataset = ImageFolder(
    root=os.path.join(dataset_path, "train"), transform=train_transform
)
val_dataset = ImageFolder(
    root=os.path.join(dataset_path, "val"), transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FingerprintLivenessCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize run_id and timestamp for logging
run_id = str(uuid.uuid4())
timestamp = datetime.now().isoformat()

# 4. Training loop
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
    header = [
        "run_id", "epoch", "timestamp", "y_true", "y_pred", "y_prob", "processing_time",
        "accuracy", "precision", "recall", "f1_score", "robustness_tnr", "bce", "source"
    ]
    csv_path = "metrics.csv"
    import pandas as pd
    epoch_df = pd.DataFrame([epoch_row])
    if os.path.exists(csv_path):
        epoch_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        epoch_df.to_csv(csv_path, header=header, index=False)

# 5. Evaluate and save metrics
all_labels = []
all_preds = []
all_probs = []
all_epochs = []
processing_times = []
run_id = str(uuid.uuid4())
timestamp = datetime.now().isoformat()

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
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
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
header = [
    "run_id", "epoch", "timestamp", "y_true", "y_pred", "y_prob", "processing_time",
    "accuracy", "precision", "recall", "f1_score", "robustness_tnr", "bce", "conf_matrix", "source"
]
if os.path.exists(csv_path):
    metrics_df.to_csv(csv_path, mode="a", header=False, index=False)
else:
    metrics_df.to_csv(csv_path, header=header, index=False)
print("Validation metrics appended to metrics.csv")

# 6. Save model
torch.save(model.state_dict(), "liveness_model.pth")
print("Model saved as liveness_model.pth")

# Calculate summary metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
# Robustness against spoofing (TNR)
import numpy as np
all_labels_np = np.array(all_labels)
all_preds_np = np.array(all_preds)
tn = ((all_labels_np == 0) & (all_preds_np == 0)).sum()
fp = ((all_labels_np == 0) & (all_preds_np == 1)).sum()
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
# Binary Cross Entropy (BCE)
if len(all_probs) > 0 and len(all_labels) > 0:
    bce = nn.BCELoss()(torch.tensor(all_probs), torch.tensor(all_labels)).item()
    if np.isnan(bce) or np.isinf(bce):
        bce = 0.0
else:
    bce = 0.0
# In summary row, add conf_matrix
summary_row = {
    'run_id': run_id,
    'epoch': 'summary',
    'timestamp': timestamp,
    'y_true': '',
    'y_pred': '',
    'y_prob': '',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'robustness_tnr': tnr,
    'bce': bce,
    'conf_matrix': ','.join(map(str, cm_flat)),
    'source': 'Standalone',
}
# Ensure all columns exist in the same order as metrics_df
all_columns = list(metrics_df.columns) + ['accuracy', 'precision', 'recall', 'f1_score', 'robustness_tnr', 'bce', 'source']
for col in all_columns:
    if col not in summary_row:
        summary_row[col] = ''
summary_row = {k: summary_row[k] for k in all_columns}
summary_df = pd.DataFrame([summary_row])
# When writing summary_df, always use the same header
if os.path.exists(csv_path):
    summary_df.to_csv(csv_path, mode="a", header=False, index=False)
else:
    summary_df.to_csv(csv_path, header=header, index=False)
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

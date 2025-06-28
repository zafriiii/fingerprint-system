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
import torch.backends.cudnn
import time

torch.backends.cudnn.benchmark = True

BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224

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
        try:
            from torchvision import models
            state_dict = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).state_dict()
            self.resnet.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
        for name, param in self.resnet.named_parameters():
            param.requires_grad = True
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
        return x 

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset_path = "data"

train_dataset = ImageFolder(
    root=os.path.join(dataset_path, "train"), transform=train_transform
)
val_dataset = ImageFolder(
    root=os.path.join(dataset_path, "val"), transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA (GPU) is not available! Please install CUDA drivers and a compatible PyTorch version.")

    print("[INFO] Available CUDA devices:")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    selected_device = None
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        if "4050" in name or "RTX 4050" in name:
            selected_device = i
            break
    if selected_device is None:
        raise RuntimeError("RTX 4050 GPU not found! Please check your drivers and CUDA setup.")
    torch.cuda.set_device(selected_device)
    device = torch.device(f"cuda:{selected_device}")
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(selected_device)} (cuda:{selected_device}), Batch size: {BATCH_SIZE}")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        print("[INFO] GPU VRAM (total/used):", result.stdout.strip())
    except Exception:
        print("[INFO] Could not query GPU VRAM with nvidia-smi.")
    model = FingerprintLivenessCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda')

    if os.path.exists("liveness_model_best.pth"):
        model.load_state_dict(torch.load("liveness_model_best.pth", map_location=device, weights_only=True))
        print("Loaded best model weights from liveness_model_best.pth")

    run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    METRICS_HEADER = [
        "run_id", "epoch", "timestamp", "y_true", "y_pred", "y_prob", "processing_time",
        "accuracy", "precision", "recall", "f1_score", "robustness_tnr", "bce", "conf_matrix", "source"
    ]

    def write_metrics_row(row, csv_path):
        import pandas as pd
        row_filled = {k: row.get(k, '') for k in METRICS_HEADER}
        df = pd.DataFrame([row_filled])
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, header=METRICS_HEADER, index=False)


    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 20

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            batch_count += 1
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_count}/{len(train_loader)}] Loss: {loss.item():.4f}"
            )
        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}"
        )

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
            "conf_matrix": '',
            "source": 'Standalone',
        }
        write_metrics_row(epoch_row, "metrics.csv")
        scheduler.step()

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(1, val_batches)
        print(f"Validation BCE: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0

            torch.save(model.state_dict(), "liveness_model_best.pth")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epochs.")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

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
            elapsed_ms = (end - start) * 1000
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(float)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_epochs.extend([NUM_EPOCHS] * len(labels))
            processing_times.extend([elapsed_ms / len(labels)] * len(labels))

    import numpy as np
    probs_tensor = torch.tensor(all_probs) if len(all_probs) > 0 else torch.tensor([])
    preds = (probs_tensor > 0.5).float().cpu().numpy() if len(all_probs) > 0 else []
    accuracy = float(accuracy_score(all_labels, preds)) if len(all_labels) > 0 else 0.0
    precision = float(precision_score(all_labels, preds, zero_division=0)) if len(all_labels) > 0 else 0.0
    recall = float(recall_score(all_labels, preds, zero_division=0)) if len(all_labels) > 0 else 0.0
    f1 = float(f1_score(all_labels, preds, zero_division=0)) if len(all_labels) > 0 else 0.0
    cm = confusion_matrix(all_labels, all_preds)
    cm_flat = cm.flatten() if cm.size == 4 else [0, 0, 0, 0]  # [TN, FP, FN, TP]
    tn = ((np.array(all_labels) == 0) & (np.array(all_preds) == 0)).sum()
    fp = ((np.array(all_labels) == 0) & (np.array(all_preds) == 1)).sum()
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    if len(all_probs) > 0 and len(all_labels) > 0:
        bce = float(nn.BCELoss()(probs_tensor, torch.tensor(all_labels)).item())
        if np.isnan(bce) or np.isinf(bce):
            bce = 0.0
    else:
        bce = 0.0
    if not (isinstance(cm_flat, (list, np.ndarray)) and len(cm_flat) == 4):
        cm_flat = [0, 0, 0, 0]

    for i in range(len(all_labels)):
        val_row = {
            "run_id": run_id,
            "epoch": NUM_EPOCHS,
            "timestamp": timestamp,
            "y_true": all_labels[i],
            "y_pred": all_preds[i],
            "y_prob": all_probs[i],
            "processing_time": processing_times[i],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "robustness_tnr": tnr,
            "bce": bce,
            "conf_matrix": ','.join(map(str, cm_flat)),
            "source": 'Standalone',
        }
        write_metrics_row(val_row, "metrics.csv")
    print("Validation metrics appended to metrics.csv")

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
    write_metrics_row(summary_row, "metrics.csv")
    print("Summary metrics appended to metrics.csv")

    def predict_single_image(image_path):
        import cv2
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA (GPU) is not available! Please install CUDA drivers and a compatible PyTorch version.")
        device = torch.device("cuda")
        model.eval()
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0

        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()
            if prob > 0.9:
                result = "Live"
            elif prob < 0.1:
                result = "Spoof"
            else:
                result = "Unknown/Non-fingerprint"
        return result

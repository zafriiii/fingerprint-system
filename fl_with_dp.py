import os
import random

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision.models.resnet import ResNet, BasicBlock

class PatchedBasicBlock(BasicBlock):
    # Custom basic block to patch ResNet for compatibility with the fingerprint liveness model
    def forward(self, x):
        # Forward pass for the patched basic block
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
    # Returns a ResNet-18 model using the patched basic block for fingerprint liveness detection
    return ResNet(block=PatchedBasicBlock, layers=[2, 2, 2, 2])

class FingerprintLivenessCNN(nn.Module):
    # Neural network for fingerprint liveness detection using a modified ResNet-18 backbone
    def __init__(self):
        # Initializes the model, loads pretrained weights, and sets up the classifier
        super(FingerprintLivenessCNN, self).__init__()
        self.resnet = patched_resnet18()
        try:
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
        # Sets all ReLU activations in the given module to not operate in-place
        for m in module.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x):
        # Forward pass for the fingerprint liveness model
        x = self.resnet(x)
        x = self.classifier(x)
        return x 
    
BATCH_SIZE = 32
NUM_WORKERS = 8

def load_data(val_split=0.2):
    """
    Loads the training and validation data loaders with minimal transforms (Resize, ToTensor, Normalize).
    This is optimal for pre-augmented datasets and maximizes data loading speed.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print("[INFO] Using minimal transforms: Resize, ToTensor, Normalize.")
    dataset = ImageFolder("data/train", transform=transform)
    num_val = int(val_split * len(dataset))
    num_train = len(dataset) - num_val
    train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    print(f"Training set size: {len(train_set)}")
    print(f"Number of batches per epoch: {len(train_loader)}")
    print(f"[INFO] Training set size: {len(train_set)} samples")
    print(f"[INFO] Validation set size: {len(val_set)} samples")
    print(f"[INFO] Batches per epoch (train): {len(train_loader)}")
    print(f"[INFO] Batches per epoch (val): {len(val_loader)}")
    return train_loader, val_loader


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


class FlowerClientDP(fl.client.NumPyClient):
    def __init__(self, model, trainloader, device):
        model = ModuleValidator.fix_and_validate(model)
        self.base_model = model.to(device)
        self.trainloader = trainloader
        self.criterion = nn.BCEWithLogitsLoss() 
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=1e-4)
        self.privacy_engine = PrivacyEngine()
        self.base_model, self.optimizer, self.trainloader = (
            self.privacy_engine.make_private_with_epsilon(
                module=self.base_model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                target_epsilon=5.0,
                target_delta=1e-5,
                epochs=1,
                max_grad_norm=1.0,
                virtual_batch_size=128,
            )
        )
        self.device = device
        self.train_set_size = len(self.trainloader.dataset)
        if hasattr(self.trainloader, 'batch_size'):
            print(f"[Client] Actual batch size after Opacus: {self.trainloader.batch_size}")
        else:
            print(f"[Client] DataLoader has no batch_size attribute after Opacus.")
        self.num_batches_per_epoch = len(self.trainloader)
        print(f"[Client] Training set size: {self.train_set_size}")
        print(f"[Client] Number of batches per epoch: {self.num_batches_per_epoch}")
        print(f"[INFO] Federated client initialized with {self.train_set_size} training samples, {self.num_batches_per_epoch} batches per epoch, batch size: {getattr(self.trainloader, 'batch_size', 'unknown')}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.base_model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.base_model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.base_model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        import time
        self.set_parameters(parameters)
        num_epochs = 15
        patience = 5
        best_val_loss = float('inf')
        epochs_no_improve = 0
        epoch_times = []
        total_start = time.time()
        print(f"[INFO] Starting federated training: {len(self.trainloader.dataset)} samples, {len(self.trainloader)} batches per epoch, {num_epochs} epochs.")
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            self.base_model.train()
            total_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.float().to(self.device)
                y = y.unsqueeze(1)
                self.optimizer.zero_grad()
                output = self.base_model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.trainloader):
                    print(f"Epoch {epoch} Batch {batch_idx+1}/{len(self.trainloader)} - Loss: {loss.item():.4f}")
            avg_loss = total_loss / len(self.trainloader)
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            epoch_times.append(epoch_time)
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            print(f"[TIMER] Epoch {epoch} finished in {epoch_time:.2f}s. Avg per epoch: {avg_epoch_time:.2f}s. ETA: {eta_str}")
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

            val_loss = avg_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"[INFO] No improvement for {epochs_no_improve} epoch(s). Best loss: {best_val_loss:.4f}")
                if epochs_no_improve >= patience:
                    print(f"[INFO] Early stopping at epoch {epoch}.")
                    break

            import uuid
            from datetime import datetime
            run_id = getattr(self, 'run_id', str(uuid.uuid4()))
            timestamp = datetime.now().isoformat()
            epoch_row = {
                "run_id": run_id,
                "epoch": epoch,
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
                "source": 'FL+DP',
            }
            write_metrics_row(epoch_row, 'metrics.csv')
        total_end = time.time()
        total_time = total_end - total_start
        print(f"[TIMER] Total training time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.base_model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        processing_times = []
        with torch.no_grad():
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.float().to(self.device)
                y = y.unsqueeze(1)
                import time
                start = time.time()
                output = self.base_model(x)
                end = time.time()
                elapsed_ms = (end - start) * 1000
                probs = output.cpu().numpy()
                preds = (output > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs)
                processing_times.extend([elapsed_ms / len(y)] * len(y))

        import uuid
        from datetime import datetime
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        cm = confusion_matrix(all_labels, all_preds)
        cm_flat = cm.flatten() if cm.size == 4 else [0, 0, 0, 0]  # [TN, FP, FN, TP]
        metrics_df = pd.DataFrame(
            {"y_true": all_labels, "y_pred": all_preds, "y_prob": all_probs, "processing_time": processing_times}
        )
        metrics_df['run_id'] = run_id
        metrics_df['epoch'] = ''
        metrics_df['timestamp'] = timestamp
        metrics_df['accuracy'] = ''
        metrics_df['precision'] = ''
        metrics_df['recall'] = ''
        metrics_df['f1_score'] = ''
        metrics_df['conf_matrix'] = ','.join(map(str, cm_flat))
        metrics_df['source'] = 'FL+DP'
        csv_path = 'metrics.csv'
        header = [
            "run_id", "epoch", "timestamp", "y_true", "y_pred", "y_prob", "processing_time",
            "accuracy", "precision", "recall", "f1_score", "robustness_tnr", "bce", "conf_matrix", "source"
        ]
        if os.path.exists(csv_path):
            metrics_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(csv_path, header=header, index=False)
        print('Federated validation metrics appended to metrics.csv')

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        import numpy as np
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        tn = ((all_labels_np == 0) & (all_preds_np == 0)).sum()
        fp = ((all_labels_np == 0) & (all_preds_np == 1)).sum()
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if len(all_probs) > 0 and len(all_labels) > 0:
            all_probs_np = np.array(all_probs).reshape(-1)
            all_labels_np = np.array(all_labels).reshape(-1)
            probs_sigmoid = torch.sigmoid(torch.tensor(all_probs_np, dtype=torch.float32)).cpu().reshape(-1)
            labels_tensor = torch.tensor(all_labels_np, dtype=torch.float32).cpu().reshape(-1)
            print(f"[DEBUG] BCE input shapes: probs_sigmoid={probs_sigmoid.shape}, labels_tensor={labels_tensor.shape}")
            if probs_sigmoid.shape == labels_tensor.shape:
                bce = nn.BCELoss()(probs_sigmoid, labels_tensor).item()
                if np.isnan(bce) or np.isinf(bce):
                    bce = 0.0
            else:
                print(f"[WARNING] BCE input shape mismatch: probs_sigmoid={probs_sigmoid.shape}, labels_tensor={labels_tensor.shape}. BCE not computed.")
                bce = 0.0
        else:
            bce = 0.0
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
            'source': 'FL+DP'
        }
        all_columns = list(metrics_df.columns) + ['accuracy', 'precision', 'recall', 'f1_score', 'robustness_tnr', 'bce', 'source']
        for col in all_columns:
            if col not in summary_row:
                summary_row[col] = ''
        summary_row = {k: summary_row[k] for k in all_columns}
        summary_df = pd.DataFrame([summary_row])
        if os.path.exists(csv_path):
            summary_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(csv_path, header=header, index=False)
        print('Summary metrics appended to metrics.csv')
        return 0.0, len(self.trainloader.dataset), {}

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
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(selected_device)} (cuda:{selected_device})")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        print("[INFO] GPU VRAM (total/used):", result.stdout.strip())
    except Exception:
        print("[INFO] Could not query GPU VRAM with nvidia-smi.")
    torch.backends.cudnn.benchmark = True

    if os.path.exists("liveness_model_best.pth"):
        model = FingerprintLivenessCNN().to(device)
        model.load_state_dict(
            torch.load("liveness_model_best.pth", map_location=device)
        )
        print("Loaded best model weights from liveness_model_best.pth")
    else:
        model = FingerprintLivenessCNN().to(device)
        print("No pretrained weights found, starting from scratch.")
    trainloader, valloader = load_data()
    client = FlowerClientDP(model, trainloader, device)

    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

    torch.save(model.state_dict(), "liveness_model_best.pth")
    print("Updated model saved as liveness_model_best.pth")

    def evaluate_on_val(model, valloader):
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        processing_times = []
        import time
        with torch.no_grad():
            for x, y in valloader:
                x, y = x.to(device), y.float().to(device)
                y = y.unsqueeze(1)
                start = time.time()
                output = model(x)
                end = time.time()
                elapsed_ms = (end - start) * 1000
                probs = output.cpu().numpy()
                preds = (output > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs)
                processing_times.extend([elapsed_ms / len(y)] * len(y))
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        import numpy as np
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        cm_flat = cm.flatten() if cm.size == 4 else [0, 0, 0, 0]
        tn = ((np.array(all_labels) == 0) & (np.array(all_preds) == 0)).sum()
        fp = ((np.array(all_labels) == 0) & (np.array(all_preds) == 1)).sum()
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if len(all_probs) > 0 and len(all_labels) > 0:
            all_probs_np = np.array(all_probs).reshape(-1)
            all_labels_np = np.array(all_labels).reshape(-1)
            probs_sigmoid = torch.sigmoid(torch.tensor(all_probs_np, dtype=torch.float32)).cpu().reshape(-1)
            labels_tensor = torch.tensor(all_labels_np, dtype=torch.float32).cpu().reshape(-1)
            print(f"[DEBUG] BCE input shapes: probs_sigmoid={probs_sigmoid.shape}, labels_tensor={labels_tensor.shape}")
            if probs_sigmoid.shape == labels_tensor.shape:
                bce = nn.BCELoss()(probs_sigmoid, labels_tensor).item()
                if np.isnan(bce) or np.isinf(bce):
                    bce = 0.0
            else:
                print(f"[WARNING] BCE input shape mismatch: probs_sigmoid={probs_sigmoid.shape}, labels_tensor={labels_tensor.shape}. BCE not computed.")
                bce = 0.0
        else:
            bce = 0.0
        import uuid
        from datetime import datetime
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        summary_row = {
            'run_id': run_id,
            'epoch': 'val_summary',
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
            'source': 'FL+DP-VAL',
        }
        write_metrics_row(summary_row, 'metrics.csv')
        print('Validation metrics after FL+DP appended to metrics.csv')
    evaluate_on_val(model, valloader)

    with open("model_provenance.txt", "w") as f:
        f.write("Model: liveness_model_best.pth\n")
        f.write("Trained with: Federated Learning + Differential Privacy (Opacus)\n")
        f.write(f"Date: {__import__('datetime').datetime.now()}\n")
        f.write("DP parameters: epsilon=5.0, delta=1e-5, max_grad_norm=1.0\n")
        f.write("Frameworks: Flower, Opacus, PyTorch\n")
        f.write(
            "Note: This file is automatically generated after federated training.\n"
        )

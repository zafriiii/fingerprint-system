import torch.backends.cudnn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torchvision.models.resnet import ResNet, BasicBlock

torch.backends.cudnn.benchmark = True  # Makes training faster on GPU

BATCH_SIZE = 128
NUM_WORKERS = 8
LEARNING_RATE = 1e-3  # Match model.py for faster learning

def load_data(val_split=0.2):
    # Set up data loading and strong augmentations for better generalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
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

# Set the column order for metrics.csv (keeps logs organized)
METRICS_HEADER = [
    "run_id", "epoch", "timestamp", "y_true", "y_pred", "y_prob", "processing_time",
    "accuracy", "precision", "recall", "f1_score", "robustness_tnr", "bce", "conf_matrix", "source"
]

def write_metrics_row(row, csv_path):
    row_filled = {k: row.get(k, '') for k in METRICS_HEADER}
    df = pd.DataFrame([row_filled])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, header=METRICS_HEADER, index=False)

def fit(self, parameters, config):
        self.set_parameters(parameters)
        num_epochs = 8
        scaler = GradScaler()
        print(f"[INFO] Starting federated training: {len(self.trainloader.dataset)} samples, {len(self.trainloader)} batches per epoch, {num_epochs} epochs.")
        for epoch in range(1, num_epochs + 1):
            self.base_model.train()
            total_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.float().to(self.device)
                y = y.unsqueeze(1)
                self.optimizer.zero_grad()
                with autocast():
                    output = self.base_model(x)
                    loss = self.criterion(output, y)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch} Batch {batch_idx+1}/{len(self.trainloader)} - Loss: {loss.item():.4f}"
                    )
                total_loss += loss.item()
            avg_loss = total_loss / len(self.trainloader)
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
            # Save BCE loss for this epoch to metrics.csv
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
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.base_model.eval()
        scaler = GradScaler()
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
                with autocast():
                    output = self.base_model(x)
                end = time.time()
                elapsed_ms = (end - start) * 1000  # Time for the batch in ms
                probs = output.cpu().numpy()
                preds = (output > 0.5).float().cpu().numpy()
                # Add confidence threshold for unknown/non-fingerprint
                for i, prob in enumerate(probs):
                    if prob > 0.9:
                        preds[i] = 1  # Live
                    elif prob < 0.1:
                        preds[i] = 0  # Spoof
                    else:
                        preds[i] = -1  # Unknown/Non-fingerprint
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs)
                processing_times.extend([elapsed_ms / len(y)] * len(y))

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

class FingerprintLivenessCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = patched_resnet18()
        try:
            from torchvision import models
            state_dict = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).state_dict()
            self.resnet.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
        for name, param in self.resnet.named_parameters():
            param.requires_grad = True  # Unfreeze all layers for full training
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1)
        )
        self._set_relu_inplace(self.resnet)
        self._set_relu_inplace(self.classifier)

    def _set_relu_inplace(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x  # No sigmoid here!
# client_dp.py (Federated client with Differential Privacy using Flower + Opacus)

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


# Model definition
class FingerprintModel(nn.Module):
    def __init__(self):
        super(FingerprintModel, self).__init__()
        base = models.resnet18(pretrained=True)
        for param in base.parameters():
            param.requires_grad = False
        base.fc = nn.Sequential(
            nn.Linear(base.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.base_model = base

    def forward(self, x):
        return self.base_model(x)


# Load a small portion of the dataset
def load_data():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder("data/train", transform=transform)
    indices = random.sample(range(len(dataset)), k=int(0.2 * len(dataset)))
    return DataLoader(Subset(dataset, indices), batch_size=16, shuffle=True)


# Flower client with DP
class FlowerClientDP(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        self.base_model = model
        self.trainloader = trainloader
        self.criterion = nn.BCELoss()

        # Make model DP-compatible
        self.base_model = ModuleValidator.fix(self.base_model)
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
            )
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.base_model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.base_model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.base_model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        num_epochs = 1  # Change this to train for more epochs
        for epoch in range(1, num_epochs + 1):
            self.base_model.train()
            total_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.trainloader):
                x, y = x, y.float()
                y = y.unsqueeze(1)
                self.optimizer.zero_grad()
                output = self.base_model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                print(
                    f"Epoch {epoch} Batch {batch_idx+1}/{len(self.trainloader)} - Loss: {loss.item():.4f}"
                )
            avg_loss = total_loss / len(self.trainloader)
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
            # Log per-epoch BCE to metrics.csv
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
            header = [
                "run_id", "epoch", "timestamp", "y_true", "y_pred", "y_prob", "processing_time",
                "accuracy", "precision", "recall", "f1_score", "robustness_tnr", "bce", "source"
            ]
            csv_path = 'metrics.csv'
            import pandas as pd
            epoch_df = pd.DataFrame([epoch_row])
            if os.path.exists(csv_path):
                epoch_df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                epoch_df.to_csv(csv_path, header=header, index=False)
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
                x, y = x, y.float()
                y = y.unsqueeze(1)
                import time
                start = time.time()
                output = self.base_model(x)
                end = time.time()
                elapsed_ms = (end - start) * 1000  # ms for the batch
                probs = output.cpu().numpy()
                preds = (output > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs)
                processing_times.extend([elapsed_ms / len(y)] * len(y))
        # Save to CSV (append per-sample metrics)
        import uuid
        from datetime import datetime
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        # After all_preds and all_labels are filled in evaluate, add confusion matrix calculation
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
        # Save summary metrics as a special row in metrics.csv
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
        # Ensure all columns exist in the same order as metrics_df
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


# Launch client
if __name__ == "__main__":
    model = FingerprintModel()
    # Load pretrained weights if available
    if os.path.exists("liveness_model.pth"):
        model.load_state_dict(
            torch.load("liveness_model.pth", map_location=torch.device("cpu"))
        )
        print("Loaded pretrained weights from liveness_model.pth")
    else:
        print("No pretrained weights found, starting from scratch.")
    trainloader = load_data()
    client = FlowerClientDP(model, trainloader)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    # Save the updated model after federated training
    torch.save(model.state_dict(), "liveness_model.pth")
    print("Updated model saved as liveness_model.pth")
    # Save provenance file
    with open("model_provenance.txt", "w") as f:
        f.write("Model: liveness_model.pth\n")
        f.write("Trained with: Federated Learning + Differential Privacy (Opacus)\n")
        f.write(f"Date: {__import__('datetime').datetime.now()}\n")
        f.write("DP parameters: epsilon=5.0, delta=1e-5, max_grad_norm=1.0\n")
        f.write("Frameworks: Flower, Opacus, PyTorch\n")
        f.write(
            "Note: This file is automatically generated after federated training.\n"
        )

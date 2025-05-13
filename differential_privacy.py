
# differential_privacy_example.py (Integrating Opacus with fingerprint CNN training)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from opacus import PrivacyEngine

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder('path_to_dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model
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
            nn.Sigmoid()
        )
        self.model = base

    def forward(self, x):
        return self.model(x)

# Training setup
model = FingerprintModel()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# Integrate Opacus
from opacus.validators import ModuleValidator
model = ModuleValidator.fix(model)
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    target_epsilon=5.0,
    target_delta=1e-5,
    epochs=10,
    max_grad_norm=1.0,
)

# Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print(f"Training complete with (ε = {privacy_engine.get_epsilon(1e-5):.2f})")

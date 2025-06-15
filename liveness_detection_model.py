
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import os

BATCH_SIZE = 32
NUM_EPOCHS = 5
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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

# 2. Dataset and transforms
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Dataset path
dataset_path = 'data'

train_dataset = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=train_transform)
val_dataset = ImageFolder(root=os.path.join(dataset_path, 'val'), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FingerprintLivenessCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

# 5. Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images).squeeze()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# 6. Save model
torch.save(model.state_dict(), "liveness_model.pth")
print("Model saved as liveness_model.pth")

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

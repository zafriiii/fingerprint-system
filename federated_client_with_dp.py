
# client_dp.py (Federated client with Differential Privacy using Flower + Opacus)

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import os
import numpy as np
import random

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
            nn.Sigmoid()
        )
        self.model = base

    def forward(self, x):
        return self.model(x)

# Load a small portion of the dataset
def load_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder("path_to_dataset/train", transform=transform)
    indices = random.sample(range(len(dataset)), k=int(0.2 * len(dataset)))
    return DataLoader(Subset(dataset, indices), batch_size=16, shuffle=True)

# Flower client with DP
class FlowerClientDP(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        self.model = model
        self.trainloader = trainloader
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Make model DP-compatible
        self.model = ModuleValidator.fix(self.model)
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            target_epsilon=5.0,
            target_delta=1e-5,
            epochs=1,
            max_grad_norm=1.0,
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for x, y in self.trainloader:
            x, y = x, y.float()
            y = y.unsqueeze(1)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
        print(f"Finished one round with ε = {self.privacy_engine.get_epsilon(1e-5):.2f}")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.trainloader.dataset), {}

# Launch client
if __name__ == "__main__":
    model = FingerprintModel()
    trainloader = load_data()
    client = FlowerClientDP(model, trainloader)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

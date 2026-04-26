from torchvision.datasets.folder import default_loader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from radiocovid.core.data import RadioCovidDataset

DATA_DIR = "../data/mini_dataset"
BATCH_SIZE = 1
EPOCHS = 1
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando dispositivo:", DEVICE)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


dataset = RadioCovidDataset(
    root=DATA_DIR,
    loader=default_loader,
    extensions=["png"],
    transform=transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(dataset.classes)
print("Clases:", dataset.classes)

model = models.vgg11(pretrained=True)
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "radiocovid_trained_mini.pth")
print("✅ Modelo MINI guardado")
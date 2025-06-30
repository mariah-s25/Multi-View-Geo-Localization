
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dualresnet import DualResNet
from Modified_Preprocessing import PairedU1652Dataset
from sklearn.metrics import accuracy_score

# --- Search space ---
LEARNING_RATES = [1e-3, 1e-4]
WEIGHT_DECAYS = [0, 1e-5]
DROPOUT_RATES = [0.0, 0.3]

# --- Setup ---
data_dir = "/content/dataset_resized"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
num_epochs = 2  # Fast tuning loop

# --- Data transforms ---
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset and use a small subset for speed
dataset = PairedU1652Dataset(root_dir=data_dir, transform_sat=transform, transform_drone=transform)
subset_size = min(len(dataset), 200)
subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])
dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)

# --- Simple evaluation ---
def evaluate(model, dataloader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x1, x2, labels in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            out1, out2, _, _ = model(x1, x2)
            avg_output = (out1 + out2) / 2
            preds.extend(avg_output.argmax(1).cpu().numpy())
            targets.extend(labels.numpy())
    return accuracy_score(targets, preds)

# --- Auto-tuning loop ---
results = []

for lr in LEARNING_RATES:
    for wd in WEIGHT_DECAYS:
        for dr in DROPOUT_RATES:
            print(f"
Testing config: lr={lr}, weight_decay={wd}, dropout={dr}")

            model = DualResNet(num_classes=100).to(device)

            # Freeze backbones
            for param in model.backbone1.parameters():
                param.requires_grad = False
            for param in model.backbone2.parameters():
                param.requires_grad = False

            # Add dropout after feature extraction
            model.classifier1 = nn.Sequential(
                nn.Dropout(dr), nn.Linear(512, 100)
            )
            model.classifier2 = nn.Sequential(
                nn.Dropout(dr), nn.Linear(512, 100)
            )

            optimizer = optim.Adam(
                list(model.classifier1.parameters()) + list(model.classifier2.parameters()),
                lr=lr, weight_decay=wd
            )
            criterion = nn.CrossEntropyLoss()

            # --- Training loop ---
            model.train()
            for epoch in range(num_epochs):
                for sat_img, drone_img, labels in dataloader:
                    sat_img, drone_img, labels = sat_img.to(device), drone_img.to(device), labels.to(device)

                    optimizer.zero_grad()
                    out1, out2, _, _ = model(sat_img, drone_img)

                    loss1 = criterion(out1, labels)
                    loss2 = criterion(out2, labels)
                    loss = (loss1 + loss2) / 2
                    loss.backward()
                    optimizer.step()

            acc = evaluate(model, dataloader)
            print(f"Config result: Accuracy = {acc:.4f}")
            results.append((lr, wd, dr, acc))

# --- Report best config ---
best = max(results, key=lambda x: x[3])
print("
Best Config:")
print(f"LR: {best[0]}, Weight Decay: {best[1]}, Dropout: {best[2]}, Accuracy: {best[3]:.4f}")


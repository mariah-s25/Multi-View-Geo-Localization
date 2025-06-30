
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dualresnet import DualResNet
from Modified_Preprocessing import PairedU1652Dataset


def fine_tune_classifier():
    # --- Config ---
    model_path = "/content/ACMMM23-Solution-MBEG/weights/best_model.pth"  # pretrained weights
    save_path = "/content/ACMMM23-Solution-MBEG/weights/finetuned_best.pth"
    data_dir = "/content/dataset_resized"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Best config from AutoTuning
    lr = 0.001
    weight_decay = 0
    dropout = 0.3
    num_epochs = 15
    batch_size = 8

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = PairedU1652Dataset(root_dir=data_dir,
                                  transform_sat=transform,
                                  transform_drone=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # --- Load model and modify classifier ---
    model = DualResNet(num_classes=100).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Freeze backbones
    for param in model.backbone1.parameters():
        param.requires_grad = False
    for param in model.backbone2.parameters():
        param.requires_grad = False

    # Apply dropout in classifier layers
    model.classifier1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 100))
    model.classifier2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 100))

    optimizer = optim.Adam(
        list(model.classifier1.parameters()) + list(model.classifier2.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    model.train()
    best_acc = 0.0

    for epoch in range(num_epochs):
        total_loss = 0
        correct1 = correct2 = total = 0

        for sat_img, drone_img, labels in dataloader:
            sat_img, drone_img, labels = sat_img.to(device), drone_img.to(device), labels.to(device)

            optimizer.zero_grad()
            out1, out2, _, _ = model(sat_img, drone_img)

            loss1 = criterion(out1, labels)
            loss2 = criterion(out2, labels)
            loss = (loss1 + loss2) / 2
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct1 += (out1.argmax(1) == labels).sum().item()
            correct2 += (out2.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc1 = correct1 / total
        acc2 = correct2 / total
        avg_acc = (acc1 + acc2) / 2
        avg_loss = total_loss / total

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Acc1: {acc1:.4f}, Acc2: {acc2:.4f} | Avg Acc: {avg_acc:.4f}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at epoch {epoch+1} (Avg Acc: {best_acc:.4f})")

    print(f"
Fine-tuning complete. Best avg accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    fine_tune_classifier()

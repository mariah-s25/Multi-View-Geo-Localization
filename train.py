
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dualresnet import DualResNet                    
from Modified_Preprocessing import PairedU1652Dataset, get_num_classes
from torch.utils.data import DataLoader
from pathlib import Path 

def train():
    # Configs
    num_epochs = get_yaml_value("num_epochs")
    lr = get_yaml_value("lr")
    batch_size = get_yaml_value("batch_size")
    data_dir = "/content/dataset_subset"
    device = torch.device("cpu")
    
    #sat_dir = '/content/dataset_subset/satellite'
    #num_classes = get_num_classes(sat_dir)
    num_classes = 100
    print(f"
 Detected number of classes: {num_classes}")

    transform_sat = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    transform_drone = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = PairedU1652Dataset(root_dir=data_dir,
                                transform_sat=transform_sat, transform_drone=transform_drone)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = DualResNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    weights_dir = "/content/ACMMM23-Solution-MBEG/weights"
    print("
 Starting training...
")
    since = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
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

        epoch_loss = total_loss / total
        acc1 = correct1 / total
        acc2 = correct2 / total
        avg_acc = (acc1 + acc2) / 2

        print(f" Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Sat Acc: {acc1:.4f}, Drone Acc: {acc2:.4f}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            weights_path = os.path.join(weights_dir, "best_model.pth")
            torch.save(model.state_dict(), weights_path)
            print(f"New best model saved with avg accuracy: {best_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"
Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s")
    print(f" Best avg accuracy: {best_acc:.4f}")
if __name__ == '__main__':
    train()


import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dualresnet import DualResNet  


def make_dataloaders(root_dir, image_size=224, batch_size=16, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ds_sat = datasets.ImageFolder(os.path.join(root_dir, "query_satellite"), transform)
    ds_dr = datasets.ImageFolder(os.path.join(root_dir, "query_drone"), transform)

    return {
        'satellite': DataLoader(ds_sat, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'drone': DataLoader(ds_dr, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }, {
        'satellite': ds_sat,
        'drone': ds_dr
    }

import numpy as np
import torch

def compute_mAP(index, good_index, junk_index):
    cmc = torch.IntTensor(len(index)).zero_()  
    if len(good_index) == 0:
        return 0.0, cmc  

    mask = ~np.isin(index, junk_index)
    index = index[mask]

    order = np.where(np.isin(index, good_index))[0]

    if len(order) == 0:
        return 0.0, cmc  

    cmc[order[0]:] = 1  

    num_good = len(good_index)
    precision_at_i = [(i + 1) / (rank + 1) for i, rank in enumerate(order)]
    ap = np.sum(precision_at_i) / num_good

    return ap, cmc

def evaluate(qf, ql, gf, gl):
    qf = qf.view(1, -1)
    scores = torch.nn.functional.cosine_similarity(gf, qf, dim=1).cpu().numpy()
    index = np.argsort(scores)[::-1]
    gl = np.asarray(gl)

    good_index = np.argwhere(gl == ql).flatten()
    junk_index = np.argwhere(gl == -1).flatten()

    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc


def extract_features(model, loader, view=1):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            if view == 1:
                feat, _, _, _ = model(imgs, torch.zeros_like(imgs))
            else:
                _, feat, _, _ = model(torch.zeros_like(imgs), imgs)
            feat = F.normalize(feat, dim=1)
            feats.append(feat.cpu())
            labels.extend(labs.numpy())
    return torch.cat(feats), np.array(labels)


def test():
    root_dir = '/content/dataset_subset_test'
    weights_path = '/content/ACMMM23-Solution-MBEG/weights/best_model.pth'

    dataloaders, datasets = make_dataloaders(root_dir)
    model = DualResNet(num_classes=len(datasets['satellite'].classes))
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model = model.to('cpu')
    model.eval()

    gf, gl = extract_features(model, dataloaders['drone'], view=2)
    qf, ql = extract_features(model, dataloaders['satellite'], view=1)

    CMCs, APs = [], []
    for i in range(len(ql)):
        ap, cmc = evaluate(qf[i], ql[i], gf, gl)
        if cmc[0] == -1:
            continue
        CMCs.append(cmc.unsqueeze(0))
        APs.append(ap)

    CMC = torch.cat(CMCs).float().mean(0).numpy()
    mAP = np.mean(APs)
    print(f"Recall@1: {CMC[0]*100:.2f}%, Recall@5: {CMC[4]*100:.2f}%, mAP: {mAP*100:.2f}%")


if __name__ == '__main__':
    test()


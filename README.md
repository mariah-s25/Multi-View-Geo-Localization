# ResNet-Based U1652 Geo-Localization (Simplified MBEG)

This repository contains a training pipeline for cross-view geo-localization on the U1652 dataset, using a Dual-ResNet-based model.

It is adapted from the original MBEG (Multi-Branch Embedding Guidance) solution presented at ACM MM 2023, but removes Local Perception Network (LPN) and uses a Dual-ResNet model for training and evaluation.

Paper Link: https://doi.acm.org/?doi=3607834.3616562

# Original Source & Attribution

This project is based on the official MBEG solution available here:

> 🔗 https://github.com/Reza-Zhu/ACMMM23-Solution-MBEG

## Code Overview:

- config: settings.yaml
- model definitions: dualresnet.py
- train: train.py 
- distillation knowledge: distill_train.py
- test: U1652_test_and_evaluate.py
- prepare dataset: Modified_Preprocessing.py
- multiply queries: multi.py
- draw heat map: draw_cam_ViT.py
- predict University160k : predict.py / predict.ipynb
- export answer.txt : export.ipynb

We specifically adapted:
- `settings.yaml`
- `Preprocessing.py` into `Modified_Preprocessing.py`
- `model_.py` into `dualresnet.py`
- `train.py`
- `U1652_test_and_evaluate.py`
---

The notebook [`ResNet_U1652.ipynb`](./ResNet_U1652.ipynb) contains the code implementation and can be used to run the training and testing.

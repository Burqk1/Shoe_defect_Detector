import cv2, torch, numpy as np
from torchvision import transforms, models
import torch.nn as nn          

import os

def load_shoe_classifier():
    mdl = models.mobilenet_v2(weights='DEFAULT')
    in_f = mdl.classifier[1].in_features
    mdl.classifier[1] = nn.Linear(in_f, 1)

    w = 'model/shoe_cls.pt'
    if os.path.exists(w):
        state = torch.load(w, map_location='cpu')
        if state['classifier.1.weight'].shape[0] == 1:     
            mdl.load_state_dict(state)
            print('[+] shoe_cls.pt yüklendi.')
        else:
            print('[UYARI] shoe_cls.pt 1000-sınıflı; lütfen yeniden eğitin!')
    else:
        print('[UYARI] shoe_cls.pt bulunamadı; önce train_shoe_cls.py çalıştırın.')
    mdl.eval();  return mdl


model = load_shoe_classifier()               

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def is_shoe(path, thr=0.6):
    img = cv2.imread(path)
    if img is None:
        return False

    logit = model(transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0))
    p = torch.sigmoid(logit).item()          

    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
    edge_ratio = np.count_nonzero(edges) / edges.size

    return (p >= thr) and (edge_ratio > 0.02)

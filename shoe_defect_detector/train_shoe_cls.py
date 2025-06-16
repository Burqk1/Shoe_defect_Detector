

import os, torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from collections import Counter

DATA_DIR = 'shoe_detector_data'   
EPOCHS   = 3
BATCH    = 32
LR       = 1e-4
DEVICE   = 'cpu'                 

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
ds = datasets.ImageFolder(DATA_DIR, tf)
print('class_to_idx :', ds.class_to_idx)       
print('label_counts :', Counter(ds.targets))   

loader = DataLoader(ds, batch_size=BATCH,
                    shuffle=True, num_workers=0,  
                    pin_memory=False)

model = models.mobilenet_v2(weights='DEFAULT')
in_f  = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_f, 1)        
criterion = nn.BCEWithLogitsLoss()

model.to(DEVICE)

opt       = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    running = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)    
        logit = model(X)
        loss  = criterion(logit, y)

        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {running/len(loader):.4f}")

os.makedirs('model', exist_ok=True)
torch.save(model.state_dict(), 'model/shoe_cls.pt')
print('✅ model/shoe_cls.pt kaydedildi')

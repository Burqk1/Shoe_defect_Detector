

import sys, os, torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from collections import Counter

EPOCHS = 3
LR     = 1e-4
POS_WT = 2.0               

for arg in sys.argv[1:]:
    if arg.startswith("EPOCHS="):
        EPOCHS = int(arg.split("=")[1])
    if arg.startswith("LR="):
        LR = float(arg.split("=")[1])
    if arg.startswith("POS_WT="):
        POS_WT = float(arg.split("=")[1])

print(f"[config]  EPOCHS={EPOCHS}  LR={LR}  POS_WT={POS_WT}")

DATA_DIR = "dataset"        
tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomApply([                
        transforms.GaussianBlur(5, (0.1, 2)),
        transforms.RandomAffine(0, scale=(0.4, 1.0))
    ], p=0.4),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

ds = datasets.ImageFolder(DATA_DIR, transform=tf)
print("class_to_idx :", ds.class_to_idx)     
print("label_counts :", Counter(ds.targets))

loader = DataLoader(ds, batch_size=32, shuffle=True,
                    num_workers=0, pin_memory=False)

model = models.mobilenet_v2(weights='DEFAULT')
in_f  = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_f, 1)    

for p in model.features.parameters():        
    p.requires_grad = False

device   = 'cuda' if torch.cuda.is_available() else 'cpu'
model    = model.to(device)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WT], device=device)
)
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

for epoch in range(EPOCHS):
    running = 0
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.float().unsqueeze(1).to(device)
        logits = model(X)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {running/len(loader):.4f}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/model.pt")
print("✅ model/model.pt kaydedildi (güncellendi)")

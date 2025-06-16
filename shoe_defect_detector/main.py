
import os, torch
from torchvision import datasets, transforms, models
print(datasets.ImageFolder('dataset', transform=transforms.ToTensor()).class_to_idx)

from torch.utils.data import DataLoader
from torch import nn, optim
from collections import Counter

DATA_DIR = "dataset"           
BATCH    = 32
EPOCHS   = 3
LR       = 1e-4                
POS_WT   = 1                  
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------------------

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

ds = datasets.ImageFolder(DATA_DIR, transform=tf)
print("class_to_idx :", ds.class_to_idx)          
print("label_counts :", Counter(ds.targets))      

loader = DataLoader(ds, batch_size=BATCH,
                    shuffle=True, num_workers=0, pin_memory=False)

model = models.mobilenet_v2(weights='DEFAULT')
in_f  = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_f, 1)          

for name, p in model.features.named_parameters():
    p.requires_grad = name.split('.')[1] in {'16', '17'}

model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WT], device=DEVICE)
)
train_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(train_params, lr=3e-4)     
for epoch in range(EPOCHS):
    running = 0
    model.train()
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)   

        logits = model(imgs)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {running/len(loader):.4f}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/model.pt")
print("✅ model/model.pt kaydedildi.")

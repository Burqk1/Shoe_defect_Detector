import sys, cv2, torch, numpy as np
from PIL import Image
from torchvision import transforms
from shoe_detector import model    

img_path = sys.argv[1] if len(sys.argv) > 1 else "images/boot.jpg"
img = cv2.imread(img_path)
if img is None:
    print("Görsel bulunamadı:", img_path)
    sys.exit(1)

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
logit = model(tf(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0))
p = torch.sigmoid(logit).item()         

edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
edge_ratio = np.count_nonzero(edges) / edges.size

print(f"p={p:.3f}   edge_ratio={edge_ratio:.3f}")

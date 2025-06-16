import sys, cv2, torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from shoe_detector import is_shoe
import numpy as np            

CNN_PATH  = "model/model.pt"
YOLO_PATH = "runs/detect/defect_v2/weights/best.pt"
TH_MINOR = 60
TH_MAJOR = 80    

def load_defect_cnn(weights=CNN_PATH):
    mdl = models.mobilenet_v2(weights=None)
    in_f = mdl.classifier[1].in_features
    mdl.classifier[1] = nn.Linear(in_f, 1)     
    mdl.load_state_dict(torch.load(weights, map_location="cpu"))
    mdl.eval()
    return mdl

cnn  = load_defect_cnn()          
yolo = YOLO(YOLO_PATH)             




def blur_score(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def preprocess(bgr):
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)

def upscale_if_small(bgr, min_side=224):

    h, w = bgr.shape[:2]
    if min(h, w) < min_side:
        scale = min_side / min(h, w)
        new_wh = (int(w * scale), int(h * scale))
        bgr = cv2.resize(bgr, new_wh, interpolation=cv2.INTER_CUBIC)
    return bgr

def unsharp(bgr):

    blur = cv2.GaussianBlur(bgr, (0, 0), 3)
    return cv2.addWeighted(bgr, 1.5, blur, -0.5, 0)

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path: str):
    img = cv2.imread(image_path)

    if img is None:
        print("Görsel bulunamadı:", image_path)
        return
    
    if img.ndim != 3 or img.shape[2] != 3:
        print("❌ Görsel RGB değil veya eksik kanallar var.")
        return
    if img.shape[0] < 100 or img.shape[1] < 100:
        print("❌ Görsel çok küçük; lütfen daha büyük bir görsel kullanın.")
        return
    if img.dtype != np.uint8:
        print("❌ Görsel uint8 değil; lütfen uint8 formatında bir görsel kullanın.")
        return
    if not is_shoe(image_path, thr=0.5):
        print("❌ Görselde ayakkabı yok / okunmuyor.")
        return

    blur = blur_score(img)
    if blur < 100:
        print(f"🔄 Görsel bulanık (blur={blur:.1f}); lütfen yeniden çekin.")
        return

    if not is_shoe(image_path, thr=0.5):
        print("❌ Görselde ayakkabı yok / okunmuyor.")
        return

    img_rgb    = preprocess(img)
    img_tensor = tf(img_rgb).unsqueeze(0)



    with torch.no_grad():
        logit = cnn(img_tensor)
        p_raw = torch.sigmoid(logit).item()    
        prob_defect = 1 - p_raw                 

    score = round(prob_defect * 100, 2)
    print(f"sigmoid(raw)={p_raw:.3f}  prob_defect={prob_defect:.3f}  Skor={score}")

    if score >= 42:
        print(f"⚠️ Majör Defect ({score}/100)")
        yolo(image_path, save=True)
    elif score <= 42:
        print(f"✅ Sağlam – Güven {100-score:.1f}/100")
    else:
        print(f"❓ Sınır durumu ({score}/100) – Manuel kontrol önerilir.")

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/defect/boot2.jpg"
    predict(img_path)

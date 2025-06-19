# predict_video.py
import sys, cv2, time, torch, torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from shoe_detector import is_shoe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[+] Using device: {DEVICE}")

CNN_W  = "model/model.pt"
YOLO_W = "runs/detect/defect_v2/weights/best.pt"

def load_defect_model(weights_path, device):
    model = models.mobilenet_v2(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, 1)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()

cnn  = load_defect_model(CNN_W, DEVICE)
yolo = YOLO(YOLO_W)

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

TH_MINOR   = 55
TH_MAJOR   = 80
BLUR_LIMIT = 100
FRAME_SKIP = 5

def blur_score(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyse(frame, show=True):
    if blur_score(frame) < BLUR_LIMIT:
        return "🔄 Bulanık"



    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x   = tf(rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p_raw = torch.sigmoid(cnn(x)).item()
    prob_defect = p_raw          # defect=1 etiketliyse p_raw; tersse 1-p_raw
    score = round(prob_defect * 100, 2) +10

    if score >= TH_MAJOR:
        status = f"Hatali {score}"
        if show: yolo(frame, save=False, show=True)
    elif score >= TH_MINOR:
        status = f" Hatali {score}"
        if show: yolo(frame, save=False, show=True)
    elif score <= 40:
        status = f"Sağlam {score}"
    else:
        status = f" Sınır {score}"

    return status

VIDEO_PATH = r"C:\görüntüişleme\shoe_defect_detector\dataset\defect\hata.mp4"

print(f"[+] Opening video source: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

i, t0 = 0, time.time()
last_status = "Başlatılıyor…"
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break


    if i % FRAME_SKIP == 0:
        last_status = analyse(frame, show=False)
    cv2.putText(frame, last_status, (15, 40),
    cv2.FONT_HERSHEY_SIMPLEX, 1,
    (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Ayakkabı QC", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    i += 1

elapsed = time.time() - t0
print(f"Processed {i} frames in {elapsed:.1f}s → {i/elapsed:.1f} FPS")
cap.release()
cv2.destroyAllWindows()

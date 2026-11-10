# Shoe Defect Detector

Ayakkabı görsellerinde ve videolarında kusur (defect) tespiti yapan bir Python / PyTorch projesi. Bir görsel üç aşamadan geçiyor: önce ayrı bir MobileNetV2 sınıflandırıcı görselde ayakkabı olup olmadığına bakıyor, sonra görselin bulanıklığı ölçülüyor, en son ikinci bir MobileNetV2 modeli kusur olasılığını hesaplıyor. Skor eşiği aşılırsa kusurlu bölgeyi işaretlemek için bir YOLOv8 modeli çalıştırılıyor. Konsol çıktıları ve arayüz metinleri Türkçe.

## Özellikler

- **Ayakkabı / ayakkabı değil kontrolü** (`shoe_detector.py`): MobileNetV2 tabanlı ikili sınıflandırıcı (`model/shoe_cls.pt`) ve Canny kenar oranı kontrolü (`edge_ratio > 0.02`).
- **Bulanıklık kontrolü**: Laplacian varyansı 100'ün altındaysa görsel reddediliyor ve yeniden çekilmesi isteniyor.
- **Kusur sınıflandırması** (`predict.py`): CLAHE ile kontrast eşitleme, 224x224'e yeniden boyutlandırma, MobileNetV2 (`model/model.pt`) ile 0–100 arası kusur skoru.
- **Kusur lokalizasyonu**: Skor 42 ve üzerindeyse YOLOv8 modeli (`runs/detect/defect_v2/weights/best.pt`) görsel üzerinde çalıştırılıyor, sonuç `runs/detect/predict*/` altına kaydediliyor.
- **Video analizi** (`predict_video.py`): Videonun her 5. karesi analiz ediliyor, durum ("Hatali", "Sağlam", "Sınır", "Bulanık") karenin üzerine yazılıyor ve OpenCV penceresinde gösteriliyor. `q` ile çıkılıyor; bitince işlenen kare sayısı ve FPS yazdırılıyor.
- **Tek tıkla çalıştırma** (`run_project.bat`): Eksik modelleri sırayla eğitip tahmin çalıştıran Windows betiği.

## Teknoloji

- Python (PyCharm proje ayarlarında Python 3.13 görünüyor)
- PyTorch, torchvision (MobileNetV2, ImageFolder)
- Ultralytics YOLOv8 (`yolov8n.pt` taban model)
- OpenCV, NumPy, Pillow
- Kaggle CLI (yalnızca `run_project.bat` içindeki veri indirme adımı için)

## Proje yapısı

```
shoe_defect_detector/
├── main.py                 # Kusur CNN'ini eğitir (MobileNetV2, son iki blok + classifier) -> model/model.pt
├── train_defect_cnn.py     # Alternatif kusur CNN eğitimi (features dondurulmuş, argüman alır) -> model/model.pt
├── train_shoe_cls.py       # Ayakkabı / ayakkabı değil sınıflandırıcısı -> model/shoe_cls.pt
├── train.py                # YOLOv8 eğitimi (dataset.yaml)
├── shoe_detector.py        # is_shoe() fonksiyonu
├── predict.py              # Tek görsel için tahmin
├── predict_video.py        # Video üzerinde tahmin
├── quick_probe.py          # Ayakkabı sınıflandırıcısının olasılığını ve kenar oranını yazdırır
├── run_project.bat         # Kurulum kontrolü + eğitim + tahmin akışı
├── dataset.yaml            # YOLO veri tanımı (tek sınıf: defect)
├── dataset/
│   ├── defect/             # Kusurlu örnekler (ImageFolder sınıfı)
│   └── normal/             # Sağlam örnekler
├── images/train, images/val  # YOLO eğitim / doğrulama görselleri
├── shoe_detector_data/
│   ├── shoe/               # Ayakkabı görselleri
│   └── not_shoe/           # Ayakkabı olmayan görseller
├── shoe_tmp/               # Kaggle'dan indirilen "Shoe vs Sandal vs Boot" veri seti
├── model/                  # model.pt, shoe_cls.pt
├── runs/                   # YOLO eğitim (train*, defect_v1, detect/defect_v2) ve tahmin çıktıları
└── yolov8n.pt              # YOLOv8n taban ağırlıkları
```

Depoda eğitilmiş ağırlıklar (`model/model.pt`, `model/shoe_cls.pt`, `runs/.../weights/best.pt`) ve Kaggle veri setinin bir kopyası (`shoe_tmp/`, yaklaşık 15.000 görsel; `shoe_detector_data/shoe/` altında 10.000 görsel) bulunuyor. Bu yüzden klon boyutu büyük.

## Gereksinimler

Kodun import ettiği paketler `shoe_defect_detector/requirements.txt` dosyasında listeleniyor (torch, torchvision, ultralytics, opencv-python, numpy, pillow):

```bash
pip install -r requirements.txt
```

Veri indirme adımı için ayrıca `pip install kaggle` ve `%USERPROFILE%\.kaggle\kaggle.json` API anahtarı gerekiyor. CUDA varsa `main.py`, `train_defect_cnn.py` ve `predict_video.py` otomatik olarak GPU kullanıyor. `train_shoe_cls.py` sabit olarak CPU'da çalışıyor.

## Kurulum

```bash
cd shoe_defect_detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Tüm betikler göreli yollar kullanıyor (`model/...`, `dataset`, `runs/...`). Bu yüzden komutları `shoe_defect_detector/` klasörünün içinden çalıştırın.

## Kullanım

### Tek görsel tahmini

```bash
python predict.py yol/gorsel.jpg
```

Argüman verilmezse `dataset/normal/boot.jpg` kullanılıyor. Görselin çözünürlüğü en az 100x100 piksel olmalı ve görsel 3 kanallı, uint8 formatında olmalı. Örnek çıktılar:

- `❌ Görselde ayakkabı yok / okunmuyor.`
- `🔄 Görsel bulanık (blur=...); lütfen yeniden çekin.`
- `⚠️ Majör Defect (skor/100)`: YOLO çalıştırılır, sonuç `runs/detect/predictN/` altına kaydedilir.
- `✅ Sağlam – Güven .../100`

### Video tahmini

```bash
python predict_video.py yol/video.mp4
```

Argüman verilmezse depodaki örnek video `dataset/defect/hata.mp4` kullanılıyor.

### Modelleri yeniden eğitme

```bash
# Ayakkabı / ayakkabı değil sınıflandırıcısı (shoe_detector_data/shoe ve not_shoe)
python train_shoe_cls.py

# Kusur CNN'i (dataset/defect ve dataset/normal)
python main.py
# veya, features katmanları dondurulmuş ve parametreli sürüm:
python train_defect_cnn.py EPOCHS=10 LR=1e-4 POS_WT=2.0

# YOLOv8 kusur dedektörü (dataset.yaml -> images/train, images/val)
python train.py --epochs 30 --imgsz 640 --batch 16 --model yolov8n.pt --name defect_v1 --exist_ok
```

`train.py` sonuçları `runs/<name>/` altına yazıyor. `predict.py` ve `predict_video.py` ise `runs/detect/defect_v2/weights/best.pt` dosyasını kullanıyor. Yeni bir YOLO modeli eğittiyseniz bu yolu güncellemeniz gerekiyor.

### Toplu akış (Windows)

```bat
run_project.bat
```

Betik şunları yapıyor:

1. `venv` klasörünün varlığını kontrol ediyor ve sanal ortamı etkinleştiriyor.
2. Kaggle CLI'yi kuruyor (yoksa) ve `kaggle.json` dosyasını kontrol ediyor.
3. `shoe_detector_data/shoe` boşsa `hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images` veri setini indiriyor. `not_shoe` klasörü boşsa duruyor; bu klasörü elle doldurmanız gerekiyor.
4. `model/shoe_cls.pt`, `model/model.pt` ve `runs/defect_v1/weights/best.pt` yoksa ilgili modeli eğitiyor.
5. `predict.py` çalıştırıyor.

## Veri

- **Kusur CNN'i** `torchvision.datasets.ImageFolder` ile `dataset/` klasörünü okuyor. Sınıf indeksleri alfabetik sırayla atanıyor (`defect=0`, `normal=1`). Bu yüzden `predict.py` kusur olasılığını `1 - sigmoid(logit)` olarak hesaplıyor. Depodaki `dataset/` altında sadece birkaç örnek dosya var; modeli yeniden eğitmek için kendi görsellerinizi eklemeniz gerekiyor.
- **YOLO** `dataset.yaml` dosyasını kullanıyor (`nc: 1`, `names: ['defect']`). `images/train` ve `images/val` klasörlerinde yalnızca birer `sample.jpg` var, etiket (`.txt`) dosyası yok.

## Notlar ve sınırlamalar

- YOLO eğitim verisi neredeyse boş olduğu için `runs/detect/defect_v2/results.csv` dosyasında precision, recall ve mAP değerleri 0. Gerçek kusur lokalizasyonu için etiketli bir YOLO veri seti hazırlanması gerekiyor.
- `predict.py` ve `predict_video.py` skoru farklı yorumluyor. Görsel tahmininde `1 - sigmoid` kullanılırken, videoda ham sigmoid değerine +10 ekleniyor ve eşikler 55/80/40. Video betiğindeki yorum satırı da bu farka işaret ediyor.
- `predict.py` içinde tanımlı `TH_MINOR` / `TH_MAJOR` sabitleri, `upscale_if_small` ve `unsharp` fonksiyonları kullanılmıyor. Fiili karar eşiği 42.
- `quick_probe.py` argüman verilmezse `images/boot.jpg` dosyasını okuyor.
- Depoda lisans dosyası yok; lisans belirtilmemiş.

# Shoe Defect Detector

A Python / PyTorch project that detects defects in shoe images and videos. An image goes through three stages: first a separate MobileNetV2 classifier checks whether the image contains a shoe, then the image's blurriness is measured, and finally a second MobileNetV2 model computes the defect probability. If the score passes the threshold, a YOLOv8 model is run to mark the defective region. Console output and UI text are in Turkish.

## Features

- **Shoe / not-shoe check** (`shoe_detector.py`): A MobileNetV2-based binary classifier (`model/shoe_cls.pt`) plus a Canny edge ratio check (`edge_ratio > 0.02`).
- **Blur check**: If the Laplacian variance is below 100, the image is rejected and the user is asked to retake it.
- **Defect classification** (`predict.py`): Contrast equalization with CLAHE, resizing to 224x224, and a defect score from 0–100 from MobileNetV2 (`model/model.pt`).
- **Defect localization**: If the score is 42 or higher, the YOLOv8 model (`runs/detect/defect_v2/weights/best.pt`) is run on the image and the result is saved under `runs/detect/predict*/`.
- **Video analysis** (`predict_video.py`): Every 5th frame of the video is analyzed, the status ("Hatali" (defective), "Sağlam" (sound), "Sınır" (borderline), "Bulanık" (blurry)) is drawn on the frame and shown in an OpenCV window. Press `q` to quit; when it finishes, the number of processed frames and the FPS are printed.
- **One-click run** (`run_project.bat`): A Windows script that trains missing models in order and then runs prediction.

## Technology

- Python (the PyCharm project settings show Python 3.13)
- PyTorch, torchvision (MobileNetV2, ImageFolder)
- Ultralytics YOLOv8 (`yolov8n.pt` base model)
- OpenCV, NumPy, Pillow
- Kaggle CLI (only for the data download step in `run_project.bat`)

## Project structure

```
shoe_defect_detector/
├── main.py                 # Trains the defect CNN (MobileNetV2, last two blocks + classifier) -> model/model.pt
├── train_defect_cnn.py     # Alternative defect CNN training (frozen features, takes arguments) -> model/model.pt
├── train_shoe_cls.py       # Shoe / not-shoe classifier -> model/shoe_cls.pt
├── train.py                # YOLOv8 training (dataset.yaml)
├── shoe_detector.py        # is_shoe() function
├── predict.py              # Prediction for a single image
├── predict_video.py        # Prediction on video
├── quick_probe.py          # Prints the shoe classifier's probability and the edge ratio
├── run_project.bat         # Setup check + training + prediction flow
├── dataset.yaml            # YOLO data definition (single class: defect)
├── dataset/
│   ├── defect/             # Defective samples (ImageFolder class)
│   └── normal/             # Sound samples
├── images/train, images/val  # YOLO training / validation images
├── shoe_detector_data/
│   ├── shoe/               # Shoe images
│   └── not_shoe/           # Non-shoe images
├── shoe_tmp/               # "Shoe vs Sandal vs Boot" dataset downloaded from Kaggle
├── model/                  # model.pt, shoe_cls.pt
├── runs/                   # YOLO training (train*, defect_v1, detect/defect_v2) and prediction outputs
└── yolov8n.pt              # YOLOv8n base weights
```

The repository contains trained weights (`model/model.pt`, `model/shoe_cls.pt`, `runs/.../weights/best.pt`) and a copy of the Kaggle dataset (`shoe_tmp/`, about 15,000 images; 10,000 images under `shoe_detector_data/shoe/`). Because of this, the clone is large.

## Requirements

The packages the code imports are listed in `shoe_defect_detector/requirements.txt` (torch, torchvision, ultralytics, opencv-python, numpy, pillow):

```bash
pip install -r requirements.txt
```

The data download step also needs `pip install kaggle` and a `%USERPROFILE%\.kaggle\kaggle.json` API key. If CUDA is available, `main.py`, `train_defect_cnn.py` and `predict_video.py` use the GPU automatically. `train_shoe_cls.py` always runs on the CPU.

## Setup

```bash
cd shoe_defect_detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

All scripts use relative paths (`model/...`, `dataset`, `runs/...`), so run the commands from inside the `shoe_defect_detector/` folder.

## Usage

### Single image prediction

```bash
python predict.py yol/gorsel.jpg
```

If no argument is given, `dataset/normal/boot.jpg` is used. The image must be at least 100x100 pixels, 3-channel, in uint8 format. Example outputs:

- `❌ Görselde ayakkabı yok / okunmuyor.` (no shoe in the image / can't be read)
- `🔄 Görsel bulanık (blur=...); lütfen yeniden çekin.` (image is blurry; please retake it)
- `⚠️ Majör Defect (skor/100)`: YOLO is run and the result is saved under `runs/detect/predictN/`.
- `✅ Sağlam – Güven .../100` (sound – confidence)

### Video prediction

```bash
python predict_video.py yol/video.mp4
```

If no argument is given, the sample video in the repository, `dataset/defect/hata.mp4`, is used.

### Retraining the models

```bash
# Shoe / not-shoe classifier (shoe_detector_data/shoe and not_shoe)
python train_shoe_cls.py

# Defect CNN (dataset/defect and dataset/normal)
python main.py
# or, the version with frozen features layers and parameters:
python train_defect_cnn.py EPOCHS=10 LR=1e-4 POS_WT=2.0

# YOLOv8 defect detector (dataset.yaml -> images/train, images/val)
python train.py --epochs 30 --imgsz 640 --batch 16 --model yolov8n.pt --name defect_v1 --exist_ok
```

`train.py` writes its results under `runs/<name>/`. `predict.py` and `predict_video.py`, however, use `runs/detect/defect_v2/weights/best.pt`. If you train a new YOLO model, you need to update this path.

### Batch flow (Windows)

```bat
run_project.bat
```

The script does the following:

1. Checks that the `venv` folder exists and activates the virtual environment.
2. Installs the Kaggle CLI (if missing) and checks for the `kaggle.json` file.
3. If `shoe_detector_data/shoe` is empty, downloads the `hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images` dataset. If the `not_shoe` folder is empty, it stops; you need to fill this folder manually.
4. Trains the corresponding model if `model/shoe_cls.pt`, `model/model.pt` or `runs/defect_v1/weights/best.pt` is missing.
5. Runs `predict.py`.

## Data

- **The defect CNN** reads the `dataset/` folder with `torchvision.datasets.ImageFolder`. Class indices are assigned alphabetically (`defect=0`, `normal=1`), which is why `predict.py` computes the defect probability as `1 - sigmoid(logit)`. The `dataset/` folder in the repository contains only a few sample files; to retrain the model you need to add your own images.
- **YOLO** uses `dataset.yaml` (`nc: 1`, `names: ['defect']`). The `images/train` and `images/val` folders contain only one `sample.jpg` each, with no label (`.txt`) files.

## Notes and limitations

- Because the YOLO training data is almost empty, precision, recall and mAP are 0 in `runs/detect/defect_v2/results.csv`. Real defect localization requires preparing a labeled YOLO dataset.
- `predict.py` and `predict_video.py` interpret the score differently. Image prediction uses `1 - sigmoid`, while the video script adds +10 to the raw sigmoid value and uses thresholds of 55/80/40. A comment in the video script also points to this difference.
- The `TH_MINOR` / `TH_MAJOR` constants and the `upscale_if_small` and `unsharp` functions defined in `predict.py` are unused. The actual decision threshold is 42.
- If no argument is given, `quick_probe.py` reads `images/boot.jpg`.
- The repository has no license file; no license is specified.

@echo off


cd /d %~dp0

if not exist venv (
    echo [HATA] venv yok. Olusturmak icin:
    echo     python -m venv venv
    echo     venv\Scripts\activate
    echo     python -m pip install -r requirements.txt
    pause & exit /b
)
call venv\Scripts\activate

echo.
echo ================= [0/4] Kurulum Kontrolu =================

where kaggle >nul 2>&1
if errorlevel 1 (
    echo Kaggle CLI bulunamadi, yukleniyor...
    python -m pip install --quiet kaggle
)

set "KJSON=%USERPROFILE%\.kaggle\kaggle.json"
if not exist "%KJSON%" (
    echo [HATA] %KJSON% bulunamadi.
    echo 1 Kaggle hesabinda: Account » Create New API Token
    echo 2 kaggle.json dosyasini bu klasore kopyala
    pause & exit /b
)

if not exist shoe_detector_data\shoe md shoe_detector_data\shoe
if not exist shoe_detector_data\not_shoe md shoe_detector_data\not_shoe

dir shoe_detector_data\shoe\*.jpg >nul 2>&1
if errorlevel 1 (
    @REM 
    if not exist shoe_tmp md shoe_tmp
    kaggle datasets download -d hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images -p shoe_tmp
    powershell -command "Expand-Archive -Path shoe_tmp\*.zip -DestinationPath shoe_tmp -Force"
    xcopy /y /q /s shoe_tmp\shoe_detector\shoe\* shoe_detector_data\shoe\
)

dir shoe_detector_data\not_shoe\*.jpg >nul 2>&1
if errorlevel 1 (
    @REM 
    pause
)

echo ==========================================================
echo.

if not exist model\shoe_cls.pt (
    echo [1/4] Shoe / Not-Shoe modeli egitiliyor...
    python train_shoe_cls.py
) else (
    echo [1/4] Shoe / Not-Shoe modeli mevcut, atlandi.
)

if not exist model\model.pt (
    echo [2/4] CNN modeli egitiliyor...
    python main.py
) else (
    echo [2/4] CNN modeli mevcut, atlandi.
)

if not exist runs\defect_v1\weights\best.pt (
    echo [3/4] YOLOv8 modeli egitiliyor...
    python train.py --epochs 30 --imgsz 640 --name defect_v1 --exist_ok
) else (
    echo [3/4] YOLOv8 modeli mevcut, atlandi.
)

echo [4/4] Tahmin yapiliyor...
python predict.py 

pause

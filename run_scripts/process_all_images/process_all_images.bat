@echo off
echo PixToTreePlan Batch Processing
echo ==========================

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo Looking for image directories...
python batch_process.py --input_dir="dataset/input_images" --visualize --max_images=5

echo.
echo If you want to process more images, run:
echo python batch_process.py --input_dir="dataset/input_images" --visualize

pause

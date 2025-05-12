@echo off
REM Batch script for running the batch processing with the new folder structure
REM This script stays in the project root and calls the appropriate scripts

echo PixToTreePlan Batch Processing
echo ============================

REM Change to the project root directory
cd /d "%~dp0"

REM Check Python installation
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.7+ and try again.
    exit /b 1
)

REM Ensure output directory exists
if not exist "outputs" mkdir outputs
if not exist "temp" mkdir temp

REM Create the symbolic link in the core directory if it doesn't exist
if not exist "core\main" (
    echo Creating symbolic link for main module...
    mklink /J "core\main" "main"
    if %errorlevel% neq 0 (
        echo Failed to create symbolic link. Using fallback method.
    )
)

REM Run the batch processing script with max_images parameter for testing first
echo Starting batch processing...
SET /p max_img="Enter maximum number of images to process (press Enter for all): "
if "%max_img%"=="" (
    python core\batch_process.py --input_dir "dataset\input_images" --extensions "jpg,png" --visualize
) else (
    python core\batch_process.py --input_dir "dataset\input_images" --extensions "jpg,png" --visualize --max_images %max_img%
)

REM Generate comparison of results
echo.
echo Generating comparison visualization...
python core\compare_results.py --image_dir "dataset\input_images" --extension "jpg" --include_pointcloud --output_path "comparison_results.png"

echo.
echo All processing complete. Results saved in outputs directory.
echo Comparison visualization saved as comparison_results.png

pause

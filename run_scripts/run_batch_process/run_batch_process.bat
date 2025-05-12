@echo off
echo PixToTreePlan Batch Processing
echo ============================

:: Check Python installation
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.7+ and try again.
    exit /b 1
)

:: Ensure output directory exists
if not exist "outputs" mkdir outputs

:: Run the batch processing script
echo Starting batch processing...
python ..\..\core\batch_process.py --input_dir "..\..\dataset\input_images" --extensions "jpg,png" --visualize

:: Generate comparison of results
echo Generating comparison visualization...
python ..\..\core\compare_results.py --image_dir "..\..\dataset\input_images" --extension "jpg" --include_pointcloud --output_path "..\..\comparison_results.png"

echo.
echo All processing complete. Results saved in output directories.
echo Comparison visualization saved as 'batch_comparison.png'
echo.
pause

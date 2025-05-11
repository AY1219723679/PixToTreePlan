@echo off
echo ===================================================
echo PixToTreePlan Output Structure Update
echo ===================================================
echo This script will update the output folder structure
echo to organize all outputs for each image in a dedicated
echo folder under 'outputs/'
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

python update_output_structure.py

echo.
echo ===================================================
echo Update process completed.
echo.
echo To test the new structure, run:
echo   python main.py
echo.
echo Press any key to exit...
pause > nul

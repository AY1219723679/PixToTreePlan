@echo off
REM This batch file creates a temporary working directory without special characters
REM and copies the necessary files there for running the demo

echo === PixToTreePlan YOLO 3D Demo ===
echo Creating temporary workspace...

REM Create a temporary directory in C:\ to avoid path problems
set TEMP_DIR=C:\PixToTreePlanDemo
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
mkdir "%TEMP_DIR%"
mkdir "%TEMP_DIR%\YOLO"
mkdir "%TEMP_DIR%\YOLO\train"
mkdir "%TEMP_DIR%\YOLO\train\images"
mkdir "%TEMP_DIR%\YOLO\train\labels"
mkdir "%TEMP_DIR%\outputs"

REM Copy the core files
set SOURCE_DIR=%~dp0
echo Copying core files from %SOURCE_DIR% to %TEMP_DIR%\YOLO...

copy "%SOURCE_DIR%\yolo_3d_compare.py" "%TEMP_DIR%\YOLO\"
copy "%SOURCE_DIR%\yolo_utils.py" "%TEMP_DIR%\YOLO\"
copy "%SOURCE_DIR%\check_depth_maps.py" "%TEMP_DIR%\YOLO\"
copy "%SOURCE_DIR%\side_by_side_viz.py" "%TEMP_DIR%\YOLO\"
copy "%SOURCE_DIR%\README_YOLO_3D.md" "%TEMP_DIR%\YOLO\"
copy "%SOURCE_DIR%\YOLO_TO_3D.md" "%TEMP_DIR%\YOLO\"
copy "%SOURCE_DIR%\data.yaml" "%TEMP_DIR%\YOLO\"

REM Copy sample images and labels
echo Copying sample images and labels...

copy "%SOURCE_DIR%\train\images\urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg" "%TEMP_DIR%\YOLO\train\images\"
copy "%SOURCE_DIR%\train\labels\urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.txt" "%TEMP_DIR%\YOLO\train\labels\"

REM Copy the corresponding outputs
echo Copying output folders...

xcopy /E /I /Y "..\..\outputs\urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab" "%TEMP_DIR%\outputs\urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab"

REM Create a simple script to run the demo
echo @echo off > "%TEMP_DIR%\run_demo.bat"
echo cd YOLO >> "%TEMP_DIR%\run_demo.bat"
echo python yolo_3d_compare.py --image "train/images/urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg" >> "%TEMP_DIR%\run_demo.bat"
echo pause >> "%TEMP_DIR%\run_demo.bat"

echo.
echo === Setup Complete ===
echo Temporary workspace created at %TEMP_DIR%
echo.
echo To run the demo:
echo 1. Navigate to %TEMP_DIR%
echo 2. Run run_demo.bat
echo.

REM Ask if user wants to open the folder
set /p OPEN_FOLDER=Open the folder now? (y/n): 
if /i "%OPEN_FOLDER%"=="y" start "" "%TEMP_DIR%"

pause

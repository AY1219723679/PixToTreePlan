@echo off
REM Check and Run YOLO 3D Compare Tool
REM This script checks if an image has matching depth maps and then runs the 
REM YOLO 3D comparison with the correct depth maps.

echo === PixToTreePlan YOLO 3D Compare Tool ===

REM Check if image parameter is provided
if "%~1"=="" (
    echo Usage: check_and_run.bat [image_path] [--auto_ground] [--force]
    echo Example: check_and_run.bat train\images\urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg
    exit /b
)

set IMAGE=%~1
set AUTO_GROUND=
set FORCE=

REM Parse additional arguments
:parse
if "%~2"=="" goto :endParse
if /i "%~2"=="--auto_ground" set AUTO_GROUND=--auto_ground
if /i "%~2"=="--force" set FORCE=--force
shift
goto :parse
:endParse

echo.
echo Checking for matching depth maps...
echo Running: python check_depth_maps.py --image "%IMAGE%"

REM Run the check_depth_maps.py script
python check_depth_maps.py --image "%IMAGE%" > check_output.tmp

REM Extract the suggested command from the output
findstr /C:"python yolo_3d_compare.py" check_output.tmp > command.tmp

REM Add any additional arguments
set CMD=
for /f "tokens=*" %%a in (command.tmp) do set CMD=%%a %AUTO_GROUND% %FORCE%

REM Show and run the command
if not "%CMD%"=="" (
    echo.
    echo Running command: %CMD%
    call %CMD%
) else (
    echo.
    echo No command found to run. Please check for errors.
    type check_output.tmp
)

REM Clean up temporary files
del check_output.tmp 2>nul
del command.tmp 2>nul

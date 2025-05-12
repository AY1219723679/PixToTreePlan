# PowerShell script to set up the symbolic link and run the demo
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath

# Create symbolic link if it doesn't exist
$mainPath = Join-Path -Path $projectRoot -ChildPath "main"
$coreMainPath = Join-Path -Path $projectRoot -ChildPath "core\main"

if (-not (Test-Path -Path $coreMainPath)) {
    Write-Host "Setting up symbolic link between core/main and main..."
    
    # Create the core directory if it doesn't exist
    $coreDir = Join-Path -Path $projectRoot -ChildPath "core"
    if (-not (Test-Path -Path $coreDir)) {
        New-Item -ItemType Directory -Path $coreDir -Force
    }
    
    # Create the symbolic link
    New-Item -ItemType Junction -Path $coreMainPath -Target $mainPath -Force
    
    Write-Host "Symbolic link created successfully."
} else {
    Write-Host "Symbolic link already exists."
}

# Run the demo script
Write-Host "Running YOLO to 3D points demo..."
python yolo_to_3d_demo.py

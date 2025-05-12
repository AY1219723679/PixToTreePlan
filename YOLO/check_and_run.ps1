#!/usr/bin/env powershell
<#
.SYNOPSIS
    Check and run YOLO 3D comparison with proper depth maps
.DESCRIPTION
    This script checks if an image has matching depth maps and then runs the YOLO 3D comparison
    with the correct depth maps. It ensures that depth maps match the image properly.
.EXAMPLE
    .\check_and_run.ps1 -image "path\to\image.jpg"
#>

param (
    [Parameter(Mandatory=$false)]
    [string]$image = "",
    
    [Parameter(Mandatory=$false)]
    [string]$label = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$auto_ground = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$force = $false
)

# Get the directory of this script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

Write-Host "=== PixToTreePlan YOLO 3D Compare Tool ==="
Write-Host "Script directory: $scriptDir"

# If no image specified, show usage and exit
if ($image -eq "") {
    Write-Host "Usage: .\check_and_run.ps1 -image 'path\to\image.jpg' [-auto_ground] [-force]"
    Write-Host "Example: .\check_and_run.ps1 -image '.\train\images\urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg'"
    exit
}

# Run the check_depth_maps.py script to find matching depth maps
Write-Host "`nChecking for matching depth maps..."
$checkCmd = "python $scriptDir\check_depth_maps.py --image `"$image`""
Write-Host "Running: $checkCmd"
$output = Invoke-Expression $checkCmd

# Parse the output to find the suggested command
$suggestedCmd = $output | Where-Object { $_ -like "*Suggested command to run:*" }
$nextLine = $false
$runCmd = ""

foreach ($line in $output) {
    if ($nextLine) {
        $runCmd = $line
        break
    }
    if ($line -like "*Suggested command to run:*") {
        $nextLine = $true
    }
}

# Add any additional arguments
if ($auto_ground) {
    $runCmd = $runCmd + " --auto_ground"
}

if ($force) {
    $runCmd = $runCmd + " --force"
}

if ($runCmd -ne "") {
    Write-Host "`nRunning the suggested command:"
    Write-Host $runCmd
    
    # Execute the command
    Invoke-Expression $runCmd
} else {
    Write-Host "`nNo suggested command found. Please check for errors above."
}

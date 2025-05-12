Write-Host "PixToTreePlan Batch Processing" -ForegroundColor Green
Write-Host "==========================" -ForegroundColor Green

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
}

Write-Host "Looking for image directories..."
python batch_process.py --input_dir="dataset/input_images" --visualize --max_images=5

Write-Host ""
Write-Host "If you want to process more images, run:" -ForegroundColor Cyan
Write-Host "python batch_process.py --input_dir='dataset/input_images' --visualize" -ForegroundColor Yellow

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

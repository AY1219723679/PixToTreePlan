# Process all images in the input_images directory
# Runs batch_process.py with default settings

# Check Python installation
$pythonCmd = "python"
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    Write-Host "Using python command"
} elseif (Get-Command "python3" -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
    Write-Host "Using python3 command"
} else {
    Write-Host "Python not found. Please install Python 3.7+ and try again."
    exit 1
}

# Ensure output directories exist
New-Item -ItemType Directory -Force -Path "output_depth" | Out-Null
New-Item -ItemType Directory -Force -Path "output_groundmasks" | Out-Null
New-Item -ItemType Directory -Force -Path "output_pointcloud" | Out-Null

# Run the batch processing script
Write-Host "Starting batch processing..."
& $pythonCmd batch_process.py --input_dir "dataset\input_images" --extensions "jpg,png" --visualize

# Generate comparison of results
Write-Host "Generating comparison visualization..."
& $pythonCmd compare_results.py --image_dir "dataset\input_images" --extension "jpg" --include_pointcloud --output_path "batch_comparison.png"

Write-Host "All processing complete. Results saved in output directories."
Write-Host "Comparison visualization saved as 'batch_comparison.png'"

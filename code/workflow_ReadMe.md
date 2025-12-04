# Workflow Runner - Image Analysis Pipeline

## Overview

This Python script processes images through a Roboflow inference workflow to detect and quantify leaf and instrument pixels in plant images. The script supports batch processing of multiple images and folders, automatically calculating area ratios and corrected area measurements.

## System Requirements

### Software Requirements
- **Python**: Version 3.9 - 3.12 (Python 3.13+ is NOT supported by inference-sdk)
- **Operating System**: Windows, macOS, or Linux
- **Inference Server**: Must be running locally on port 9001

### Python Dependencies
- `inference-sdk` - Roboflow inference client library
- Built-in libraries: `sys`, `os`, `csv`, `pathlib`

## Installation

### Step 1: Install Python 3.11
1. Download Python 3.11 from: https://www.python.org/downloads/release/python-31110/
2. During installation, **check the box "Add Python to PATH"**
3. Restart your terminal/PowerShell after installation

### Step 2: Install Required Package
```bash
py -3.11 -m pip install inference-sdk
```

### Step 3: Verify Installation
```bash
py -3.11 --version
```
Should display: `Python 3.11.x`

## Roboflow Model Details

### Workflow Information
- **Workspace**: `mastersthesis-d11wq`
- **Workflow ID**: `detect-count-and-visualize-5`
- **API URL**: `http://localhost:9001`
- **API Key**: `g76oBLBIPwupyLnfl0S0`

### Model Functionality
The workflow performs instance segmentation to detect and segment:
- **Leaf/Needle pixels**: Plant material to be measured
- **Instrument pixels**: Reference object for scale calibration

The model outputs polygon coordinates for each detected object, with each point representing a pixel in the segmented region.

### Detection Output Structure
The model returns predictions in the following format:
```json
{
  "predictions": {
    "predictions": [
      {
        "class": "leaf",
        "confidence": 0.92,
        "points": [{"x": 100, "y": 200}, ...],
        "detection_id": "uuid"
      }
    ]
  }
}
```

## Usage

### Prerequisites
Ensure your local Roboflow inference server is running on `http://localhost:9001` before executing the script.

### Basic Commands

#### Process a Single Image
```bash
py -3.11 workflow_runner.py "path/to/image.jpg"
```

#### Process All Images in a Folder
```bash
py -3.11 workflow_runner.py "path/to/folder"
```

#### Process All Images in Folder and Subfolders (Recursive)
```bash
py -3.11 workflow_runner.py "path/to/folder" --recursive
```

### Example Commands

**Windows:**
```powershell
# Single image
py -3.11 workflow_runner.py "C:/Users/haigh/OneDrive/Documents/Masters/Roboflow/Images/7-16_tree119_5461_day14/20250731_124130.jpg"

# Single folder
py -3.11 workflow_runner.py "C:/Users/haigh/OneDrive/Documents/Masters/Roboflow/Images/7-16_tree119_5461_day14"

# All subfolders
py -3.11 workflow_runner.py "C:/Users/haigh/OneDrive/Documents/Masters/Roboflow/Images" --recursive
```

**macOS/Linux:**
```bash
# Single image
python3.11 workflow_runner.py "/Users/username/Images/sample.jpg"

# Recursive processing
python3.11 workflow_runner.py "/Users/username/Images" --recursive
```

## Output

### CSV File Structure

The script generates `workflow_results.csv` with the following columns:

| Column | Description | Calculation |
|--------|-------------|-------------|
| `subfolder` | Name of the subfolder containing the image | Extracted from path |
| `image_name` | Filename only (e.g., "image001.jpg") | `os.path.basename()` |
| `image_path` | Full path to the image file | Original file path |
| `status` | Processing status ("success" or "failed") | - |
| `instrument_pixels` | Total pixels detected as "instrument" class | Count of polygon points |
| `needle_pixels` | Total pixels detected as "leaf" or "needle" class | Count of polygon points |
| `area_ratio` | Proportion of needle pixels to total pixels | `needle_pixels / (needle_pixels + instrument_pixels)` |
| `corrected_area` | Scaled area measurement in cm² | `area_ratio * 6` |

### Sample Output
```csv
subfolder,image_name,image_path,status,instrument_pixels,needle_pixels,area_ratio,corrected_area
7-16_tree119_5461_day14,20250731_124130.jpg,C:/Images/7-16_tree119_5461_day14/20250731_124130.jpg,success,15420,8234,0.3481,2.0886
7-16_tree119_5461_day14,20250731_124610.jpg,C:/Images/7-16_tree119_5461_day14/20250731_124610.jpg,success,14892,7856,0.3453,2.0718
```

### Output File Location
The CSV file is saved in the same directory where the script is executed. If the file is locked (open in Excel), the script will automatically create numbered versions:
- `workflow_results_1.csv`
- `workflow_results_2.csv`
- etc.

## Supported Image Formats

The script processes images with the following extensions:
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`, `.tif`
- `.gif`

## Script Behavior

### Processing Flow
1. **Image Discovery**: Scans the specified directory for supported image files
2. **Workflow Execution**: Sends each image to the Roboflow inference server
3. **Pixel Counting**: Extracts polygon points from detection results
4. **Calculation**: Computes area ratio and corrected area
5. **CSV Export**: Saves results with subfolder identification

### Error Handling
- **Missing Images**: Script reports file not found errors
- **Server Connection**: Displays connection errors if inference server is unavailable
- **Failed Detections**: Marks images as "failed" but continues processing remaining images
- **File Locking**: Automatically creates alternate CSV filenames if output file is locked

### Caching
The workflow uses `use_cache=True` to speed up repeated requests on the same images.

## Calculations Explained

### Area Ratio
The area ratio represents the proportion of plant material (needles/leaves) relative to the total detected area:

```
area_ratio = needle_pixels / (needle_pixels + instrument_pixels)
```

This normalized metric ranges from 0 to 1 and is independent of image size or distance from camera.

### Corrected Area
The corrected area scales the area ratio by a calibration factor of 6 cm²:

```
corrected_area = area_ratio × 6
```

This assumes the instrument provides a 6 cm² reference area for scale calibration.

## Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'inference_sdk'`  
**Solution**: Install the package:
```bash
py -3.11 -m pip install inference-sdk
```

**Problem**: `ERROR: No matching distribution found for inference-sdk`  
**Solution**: Verify you're using Python 3.9-3.12 (NOT 3.13+)

**Problem**: API connection errors  
**Solution**: Ensure the inference server is running on `http://localhost:9001`

**Problem**: `Permission denied: 'workflow_results.csv'`  
**Solution**: Close the CSV file in Excel or other programs. The script will auto-generate alternate filenames.

**Problem**: No images found  
**Solution**: 
- Verify the path is correct
- Check that images have supported extensions (.jpg, .png, etc.)
- Use `--recursive` flag for subfolder scanning

**Problem**: All detections show 0 pixels  
**Solution**: 
- Check that the inference server is returning predictions
- Verify the workflow is properly configured
- Ensure images contain detectable objects

## Script Metadata

- **Version**: 1.0
- **Author**: Created for Masters Thesis Research
- **Purpose**: Automated leaf area measurement using computer vision
- **Last Updated**: 2025
- **License**: Research/Academic Use

## File Structure

```
project/
│
├── workflow_runner.py          # Main script
├── README.md                   # This file
└── workflow_results.csv        # Generated output (after running)
```

## Additional Notes

### Performance Considerations
- Processing speed depends on image size and server performance
- Typical processing: 1-3 seconds per image
- Batch processing displays progress: `[5/30]` indicates image 5 of 30

### Data Integrity
- The script preserves original image files (read-only access)
- All calculations are performed in-memory
- CSV output uses UTF-8 encoding for compatibility

### Reproducibility
- Using `use_cache=True` ensures consistent results for the same images
- The script includes debug output on first run showing detection structure
- Timestamp information is preserved in the original image filenames

## Support and Contributions

For issues related to:
- **Roboflow Model**: Contact the model creator or check Roboflow documentation
- **Inference Server**: See Roboflow inference server documentation
- **Script Functionality**: Review error messages and troubleshooting section above

## Citation

If using this script in research, please cite appropriately and acknowledge the use of Roboflow's inference platform.

---

**Note**: This script is designed for research purposes. Ensure compliance with your institution's data management and research ethics policies.
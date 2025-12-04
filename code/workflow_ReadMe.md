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

### Output Structure
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

## Use

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


### Output File Location
The CSV file is saved in the same directory where the script is executed. If the file is locked (open in Excel), the script will automatically create numbered versions:
- `workflow_results_1.csv`
- `workflow_results_2.csv`
- etc.


## Calculations 

### Area Ratio
The area ratio represents the proportion of plant material (needles/leaves) relative to the total detected area:

```
area_ratio = needle_pixels / (needle_pixels + instrument_pixels)
```


### Corrected Area
The corrected area scales the area ratio by a calibration factor of 6 cm²:

```
corrected_area = area_ratio × 6
```

This assumes the instrument provides a 6 cm² reference area for scale calibration.

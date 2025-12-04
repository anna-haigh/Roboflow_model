# Model Evaluation

## Overview

This evaluation assesses Roboflow YOLO segmentation model's performance using:
- **Precision-Recall curves** - Detection performance across confidence thresholds
- **F1 Score optimization** - Find the optimal confidence threshold
- **Confusion Matrix** - Classification accuracy at optimal threshold  
- **IoU Distribution** - Pixel-level segmentation quality
- **Performance metrics** - Detailed statistics for model tuning

## Software Requirements
- Python 3.9-3.12 (NOT 3.13+)
- Roboflow inference server running on `http://localhost:9001`
- Ground truth annotations (test set from Roboflow)

## Step-by-Step Evaluation 

### Step 1: Download Ground Truth Annotations

Run the download script:
```bash
py -3.11 download_annotations.py
```
Enter when prompted:
   - API key
   - Workspace ID: `mastersthesis-d11wq`
   - Project ID: (your project name)
   - Version: (dataset version number, e.g., `1`)

**Result:** Annotations saved to `ground_truth/annotations.json`

### Step 2: Prepare Test Images

1. Create a folder with your test images (images NOT used for training):
```
test_images/
├── image001.jpg
├── image002.jpg
├── image003.jpg
└── ...
```

### Step 3: Run Evaluation

```bash
py -3.11 model_evaluation.py test_images/
```

The script will:
1. Run inference on all test images
2. Match predictions to ground truth using IoU
3. Calculate metrics at different confidence thresholds
4. Generate visualizations
5. Save results to `evaluation_results/`

## Output Files

After running, you'll find these files in `evaluation_results/`:

### 1. `precision_recall_curve.png`
- Shows model's precision vs recall tradeoff
- Higher AUC (area under curve) = better model
- **How to read:** Top-right corner is ideal (high precision AND high recall)

### 2. `f1_vs_threshold.png`
- Shows F1 score at each confidence threshold
- Red dashed line indicates optimal threshold
- **How to use:** Set your model's confidence threshold to the optimal value

### 3. `confusion_matrix.png`
- Shows True Positives, False Positives, False Negatives
- Evaluated at the optimal threshold
- **How to read:** Diagonal = correct predictions, off-diagonal = errors

### 4. `iou_distribution.png`
- Shows segmentation quality (IoU) vs confidence threshold
- Higher IoU = better pixel-level segmentation
- **How to read:** Values > 0.5 are generally considered good matches

### 5. `metrics_summary.txt`
- Text file with detailed statistics
- Includes metrics at all thresholds
- **Use for:** Reporting results, comparing model versions

## Understanding the Metrics

### Detection Metrics

| Metric | Formula | What it Means | Good Value |
|--------|---------|---------------|------------|
| **Precision** | TP / (TP + FP) | Of all detections, what % are correct? | >0.80 |
| **Recall** | TP / (TP + FN) | Of all objects, what % did we find? | >0.80 |
| **F1 Score** | 2 × (P × R) / (P + R) | Balanced measure of precision & recall | >0.80 |

### Segmentation Metrics

| Metric | What it Measures | Good Value |
|--------|------------------|------------|
| **IoU (Intersection over Union)** | Pixel-level segmentation accuracy | >0.50 |
| **Average IoU** | Mean IoU across all detections | >0.60 |

### Definitions

- **True Positive (TP)**: Correctly detected object with IoU ≥ 0.5
- **False Positive (FP)**: Detected object with no matching ground truth OR IoU < 0.5
- **False Negative (FN)**: Ground truth object that wasn't detected

## Interpreting Results

### Good Model Performance
```
Precision: >0.80 (few false detections)
Recall: >0.80 (finds most objects)
F1 Score: >0.80 (balanced performance)
Average IoU: >0.60 (accurate segmentation)
```

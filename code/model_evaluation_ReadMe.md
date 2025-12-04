# Model Evaluation Guide

## Overview

This evaluation suite assesses your Roboflow YOLO segmentation model's performance using:
- **Precision-Recall curves** - Detection performance across confidence thresholds
- **F1 Score optimization** - Find the optimal confidence threshold
- **Confusion Matrix** - Classification accuracy at optimal threshold  
- **IoU Distribution** - Pixel-level segmentation quality
- **Performance metrics** - Detailed statistics for model tuning

## Prerequisites

### Software Requirements
- Python 3.9-3.12 (NOT 3.13+)
- Roboflow inference server running on `http://localhost:9001`
- Ground truth annotations (test set from Roboflow)

### Additional Python Packages
```bash
py -3.11 -m pip install inference-sdk numpy matplotlib seaborn scikit-learn pillow requests
```

## Step-by-Step Evaluation Process

### Step 1: Download Ground Truth Annotations

You have two options to get your ground truth data:

#### Option A: Download via Roboflow API

1. Get your Roboflow API key:
   - Go to https://app.roboflow.com/
   - Navigate to Settings → Roboflow API
   - Copy your API key

2. Run the download script:
```bash
py -3.11 download_annotations.py
```

3. Enter when prompted:
   - API key
   - Workspace ID: `mastersthesis-d11wq`
   - Project ID: (your project name)
   - Version: (dataset version number, e.g., `1`)

#### Option B: Export from Roboflow UI

1. In Roboflow, go to your dataset version
2. Click "Export" → "COCO JSON" → Download
3. Save the ZIP file
4. Run:
```bash
py -3.11 download_annotations.py
```
5. Select option 2 and provide the ZIP file path

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

2. **Important:** Test images must be from your Roboflow test split or validation split
3. Ensure image filenames match those in your annotations

### Step 3: Update Evaluation Script

Edit `model_evaluation.py` to load your annotations:

```python
# Around line 220, replace:
ground_truth_data = {}

# With:
import json
with open('ground_truth/annotations.json', 'r') as f:
    ground_truth_data = json.load(f)
```

### Step 4: Run Evaluation

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

### Common Issues and Solutions

#### Low Precision (many false positives)
- **Problem:** Model detects objects that aren't there
- **Solution:** Increase confidence threshold
- **Or:** Add more negative examples to training data

#### Low Recall (many false negatives)
- **Problem:** Model misses objects
- **Solution:** Decrease confidence threshold
- **Or:** Add more training examples of missed cases

#### Low IoU (poor segmentation)
- **Problem:** Bounding boxes/masks don't align well with objects
- **Solution:** Improve annotation quality
- **Or:** Use a larger backbone model
- **Or:** Increase training epochs

#### Precision-Recall Tradeoff
- Higher confidence threshold → Higher precision, lower recall
- Lower confidence threshold → Higher recall, lower precision
- **Sweet spot:** F1 score maximum (shown in plots)

## Advanced Usage

### Evaluate at Specific Threshold

Modify the confidence threshold in the script:

```python
# In model_evaluation.py, change:
confidence_thresholds = np.arange(0.1, 1.0, 0.05)

# To evaluate only at 0.5:
confidence_thresholds = [0.5]
```

### Per-Class Evaluation

To evaluate "instrument" and "leaf" separately, modify the matching function to group by class before calculating metrics.

### Custom IoU Threshold

Change the IoU matching threshold:

```python
# In model_evaluation.py, change:
matches, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
    filtered_preds, ground_truth, iou_threshold=0.5  # Change this value
)
```

Common values:
- 0.5 - Standard for COCO dataset
- 0.75 - Strict matching
- 0.3 - Lenient matching

## Troubleshooting

### "No module named 'PIL'"
```bash
py -3.11 -m pip install pillow
```

### "No module named 'sklearn'"
```bash
py -3.11 -m pip install scikit-learn
```

### Images not matching annotations
- Verify image filenames match exactly (including extensions)
- Check that test images are in your annotations file
- Ensure you downloaded the correct dataset version

### All IoU values are 0
- Check that ground truth has segmentation polygons, not just bounding boxes
- Verify image dimensions are correct in the script
- Update `image_width` and `image_height` in `calculate_iou_polygon()`

### Server connection errors
- Ensure inference server is running: `http://localhost:9001`
- Check API key is correct
- Verify workspace and workflow IDs are accurate

## Workflow Integration

### Compare Model Versions

1. Run evaluation on version 1:
```bash
py -3.11 model_evaluation.py test_images/
mv evaluation_results evaluation_results_v1
```

2. Update model to version 2, then:
```bash
py -3.11 model_evaluation.py test_images/
mv evaluation_results evaluation_results_v2
```

3. Compare F1 scores and IoU values to determine which version performs better

### Set Optimal Threshold in Production

After evaluation, update your workflow runner to use the optimal threshold:

```python
# In workflow_runner.py, filter predictions:
optimal_threshold = 0.65  # From evaluation results

filtered_predictions = [
    pred for pred in predictions 
    if pred.get('confidence', 0) >= optimal_threshold
]
```

## Citation and References

### Metrics Based On
- **COCO Dataset Evaluation Metrics**: https://cocodataset.org/#detection-eval
- **YOLO Model Evaluation**: https://docs.ultralytics.com/guides/yolo-performance-metrics/

### Roboflow Documentation
- Model Evaluation: https://docs.roboflow.com/
- API Reference: https://docs.roboflow.com/api-reference/

## Support

For issues:
1. Check that all dependencies are installed
2. Verify ground truth annotations loaded correctly
3. Ensure test images match annotation filenames
4. Review error messages for specific issues

---

**Note:** This evaluation framework is designed for object detection and instance segmentation models. Results provide quantitative metrics for model comparison and optimization.
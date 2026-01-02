#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates Roboflow model performance with Precision-Recall curves,
confusion matrices, and pixel-level segmentation metrics.
Uses LOCAL inference server with DIRECT model inference.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from inference_sdk import InferenceHTTPClient
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
import requests


def setup_client():
    """Initialize and return the InferenceHTTPClient for LOCAL server"""
    print("Connecting to LOCAL inference server at http://localhost:9001...")
    client = InferenceHTTPClient(
        api_url="http://localhost:9001",
        api_key="g76oBLBIPwupyLnfl0S0"
    )
    print("Connected successfully!")
    return client

def load_ground_truth_annotations(annotations_file='LI6800_image_correction.v12i.coco-segmentation/test/_annotations.coco.json'):
    """
    Load ground truth annotations from file

    Args:
    annotations_file: Path to annotations JSON file

    Returns:
    Dictionary mapping image names to annotations
    """
    if not os.path.exists(annotations_file):
        print(f"Warning: Ground truth file not found: {annotations_file}")
        return{}

    with open(annotations_file, 'r') as f:
        ground_truth = json.load(f)

        print(f"Loaded ground truth for {len(ground_truth)} images")
        return ground_truth

def get_model_id():
    """Get the model ID - either from environment or prompt user"""
    # Try to auto-detect first
    workspace = "mastersthesis-d11wq"
    
    # Common patterns
    possible_models = [
        f"{workspace}/li6800-image-correction/12",
        f"{workspace}/li6800/12",
        f"{workspace}/instrument-detection/12",
    ]
    
    print("\nAttempting to auto-detect model...")
    print(f"Workspace: {workspace}")
    print("\nIf auto-detection fails, find your model ID at:")
    print("  Roboflow.com → Your Project → Deploy tab")
    print("  Format: workspace/project-name/version")
    
    return None  # Will be set by user input if needed


def calculate_iou_polygon(pred_points, gt_points, image_width, image_height):
    """
    Calculate IoU between predicted and ground truth polygons
    
    Args:
        pred_points: List of predicted polygon points
        gt_points: List of ground truth polygon points
        image_width: Image width
        image_height: Image height
    
    Returns:
        IoU score (0-1)
    """
    try:
        from PIL import Image, ImageDraw
        
        # Create binary masks
        pred_mask = Image.new('L', (image_width, image_height), 0)
        gt_mask = Image.new('L', (image_width, image_height), 0)
        
        # Draw polygons
        pred_draw = ImageDraw.Draw(pred_mask)
        gt_draw = ImageDraw.Draw(gt_mask)
        
        pred_poly = [(p['x'], p['y']) for p in pred_points]
        gt_poly = [(p['x'], p['y']) for p in gt_points]
        
        pred_draw.polygon(pred_poly, fill=255)
        gt_draw.polygon(gt_poly, fill=255)
        
        # Convert to numpy arrays
        pred_array = np.array(pred_mask) > 0
        gt_array = np.array(gt_mask) > 0
        
        # Calculate IoU
        intersection = np.logical_and(pred_array, gt_array).sum()
        union = np.logical_or(pred_array, gt_array).sum()
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return iou
        
    except Exception as e:
        print(f"Warning: Error calculating IoU: {e}")
        return 0.0


def match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold=0.5):
    """Match predictions to ground truth using IoU
    
    Args:
    predictions: List of prediction dictionaries
    ground_truth: List of ground truth dictionaries
    iou_threshold: Minimum IoU to consider a match
    
    Returns:
    Tuple of (matches, unmatches_predictions, unmatches_ground_truth)
    """
    matches = []
    matched_gt_indices = set()
    unmatched_predictions = []
    
    image_width=2000
    image_height=3000

    for pred_idx, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt_indices:
                continue
            
            # Only match same class
            pred_class = pred.get('class', '').lower()
            gt_class = gt.get('class', '').lower()
            
            # Handle 'leaf' vs 'needle' naming
            if pred_class == 'leaf':
                pred_class = 'needle'
            if gt_class == 'leaf':
                gt_class = 'needle'
            
            if pred_class != gt_class:
                continue
            
            # Calculate IoU
            iou = calculate_iou_polygon(
                pred.get('points', []),
                gt.get('points', []),
                image_width,
                image_height
            )
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            matches.append({
                'prediction': pred,
                'ground_truth': ground_truth[best_gt_idx],
                'iou': best_iou,
                'confidence': pred.get('confidence', 1.0)
            })
            matched_gt_indices.add(best_gt_idx)
        else:
            unmatched_predictions.append(pred)
    
    unmatched_ground_truth = [
        gt for idx, gt in enumerate(ground_truth) 
        if idx not in matched_gt_indices
    ]
    
    return matches, unmatched_predictions, unmatched_ground_truth


def run_inference_direct(client, image_path, model_id):
    """
    Run inference directly on the model (not through workflow)
    Tries multiple model ID formats automatically
    
    Args:
        client: InferenceHTTPClient instance
        image_path: Path to image file
        model_id: Model ID string (can be workspace/project/version)
    
    Returns:
        Predictions dictionary
    """
    # Generate alternative formats to try
    formats_to_try = [model_id]  # Start with user-provided format
    
    # If full path provided (workspace/project/version), try without workspace
    if model_id.count('/') == 2:
        parts = model_id.split('/')
        formats_to_try.append(f"{parts[1]}/{parts[2]}")  # project/version
    
    # Try project-version format
    if '/' in model_id:
        parts = model_id.split('/')
        project = parts[-2] if len(parts) >= 2 else parts[0]
        version = parts[-1]
        formats_to_try.append(f"{project}-v{version}")
    
    # Try each format
    last_error = None
    for format_to_try in formats_to_try:
        try:
            print(f"    Trying format: {format_to_try}")
            result = client.infer(image_path, model_id=format_to_try)
            print(f"    ✓ Success with: {format_to_try}")
            return result
        except Exception as e:
            last_error = e
            continue
    
    # If all formats failed, raise the last error
    raise last_error


def evaluate_model(client, test_images_dir, ground_truth_data, model_id, confidence_thresholds=None):
    """Evaluate model performance across different confidence thresholds
    
    Args:
        client: InferenceHTTPClient instance
        test_images_dir: Directory containing test images
        ground_truth_data: Ground truth annotations
        model_id: Model ID for inference
        confidence_thresholds: List of confidence thresholds to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    import base64
    
    if confidence_thresholds is None:
        confidence_thresholds = np.arange(0.1, 1.0, 0.05)
    
    print("\nEvaluating model performance...")
    print(f"Model ID: {model_id}")
    print("="*50)
    
    all_predictions = []
    all_ground_truths = []
    results_by_threshold = defaultdict(lambda: {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'total_iou': 0.0,
        'matched_count': 0
    })
    
    # Get list of test images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(Path(test_images_dir).glob(ext)))
    
    print(f"Processing {len(image_files)} test images...\n")
    
    successful_inferences = 0
    failed_inferences = 0
    
    for img_idx, image_path in enumerate(image_files, 1):
        print(f"[{img_idx}/{len(image_files)}] {image_path.name}")
        
        try:
            # FIX: Read and encode image as base64
            with open(str(image_path), "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Run inference with base64 encoded image
            result = client.infer(
                image_data,
                model_id=model_id
            )
            
            # Extract predictions - direct format from infer()
            predictions = result.get('predictions', [])
            
            print(f"  Detected: {len(predictions)} objects")
            successful_inferences += 1
            
        except Exception as e:
            print(f"  ERROR: {e}")
            failed_inferences += 1
            continue
        
        # Get ground truth for this image
        image_name = image_path.name
        ground_truth = ground_truth_data.get(image_name, [])
        print(f"  Ground truth: {len(ground_truth)} objects")

        for pred in predictions:
            all_predictions.append({
                'confidence': pred.get('confidence', 1.0),
                'class': pred.get('class'),
                'image': image_name
            })

        for gt in ground_truth:
            all_ground_truths.append({
                'class': gt.get('class'),
                'image': image_name
            })
        
        if not ground_truth:
            print(f"  WARNING: No ground truth found for {image_name}")
            continue
        
        # Evaluate at each threshold
        for threshold in confidence_thresholds:
            # Filter predictions by confidence
            filtered_preds = [
                p for p in predictions 
                if p.get('confidence', 1.0) >= threshold
            ]
            
            # Match predictions to ground truth
            matches, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
                filtered_preds, ground_truth
            )
            
            # Update metrics
            results_by_threshold[threshold]['true_positives'] += len(matches)
            results_by_threshold[threshold]['false_positives'] += len(unmatched_preds)
            results_by_threshold[threshold]['false_negatives'] += len(unmatched_gts)
            
            for match in matches:
                results_by_threshold[threshold]['total_iou'] += match['iou']
                results_by_threshold[threshold]['matched_count'] += 1
    
    print(f"\n{'='*50}")
    print(f"Inference summary:")
    print(f"  Successful: {successful_inferences}")
    print(f"  Failed: {failed_inferences}")
    print(f"{'='*50}")
    
    if successful_inferences == 0:
        raise Exception("No successful inferences! Check your model ID and local inference server.")
    
    print("\nCalculating metrics across thresholds...")
    
    # Calculate precision, recall, F1 for each threshold
    metrics = []
    for threshold in sorted(confidence_thresholds):
        tp = results_by_threshold[threshold]['true_positives']
        fp = results_by_threshold[threshold]['false_positives']
        fn = results_by_threshold[threshold]['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_iou = (results_by_threshold[threshold]['total_iou'] / 
                   results_by_threshold[threshold]['matched_count'] 
                   if results_by_threshold[threshold]['matched_count'] > 0 else 0)
        
        metrics.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou': avg_iou,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    return {
        'metrics_by_threshold': metrics,
        'all_predictions': all_predictions,
        'all_ground_truths': all_ground_truths
    }


def plot_precision_recall_curve(metrics, output_dir='evaluation_results'):
    """Plot Precision-Recall curve"""
    os.makedirs(output_dir, exist_ok=True)
    
    precisions = [m['precision'] for m in metrics]
    recalls = [m['recall'] for m in metrics]
    
    # Sort by recall for proper curve
    sorted_pairs = sorted(zip(recalls, precisions))
    recalls_sorted = [r for r, p in sorted_pairs]
    precisions_sorted = [p for r, p in sorted_pairs]
    
    # Calculate AUC
    if len(recalls_sorted) > 1:
        pr_auc = auc(recalls_sorted, precisions_sorted)
    else:
        pr_auc = 0
    
    plt.figure(figsize=(10, 8))
    plt.plot(recalls_sorted, precisions_sorted, 'b-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    output_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_f1_vs_threshold(metrics, output_dir='evaluation_results'):
    """Plot F1 score vs confidence threshold"""
    os.makedirs(output_dir, exist_ok=True)
    
    thresholds = [m['threshold'] for m in metrics]
    f1_scores = [m['f1'] for m in metrics]
    
    # Find optimal threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'g-', linewidth=2)
    plt.axvline(best_threshold, color='r', linestyle='--', 
                label=f'Optimal = {best_threshold:.2f} (F1 = {best_f1:.3f})')
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Confidence Threshold', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    output_path = os.path.join(output_dir, 'f1_vs_threshold.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return best_threshold, best_f1


def plot_confusion_matrix(metrics, optimal_threshold, output_dir='evaluation_results'):
    """Plot confusion matrix at optimal threshold"""
    os.makedirs(output_dir, exist_ok=True)
    
    optimal_metrics = next((m for m in metrics if abs(m['threshold'] - optimal_threshold) < 0.01), None)
    
    if optimal_metrics is None:
        print("Warning: Could not find metrics at optimal threshold")
        return
    
    tp = optimal_metrics['tp']
    fp = optimal_metrics['fp']
    fn = optimal_metrics['fn']
    tn = 0
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.2f})', 
              fontsize=14, fontweight='bold')
    
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_iou_distribution(metrics, output_dir='evaluation_results'):
    """Plot IoU distribution across thresholds"""
    os.makedirs(output_dir, exist_ok=True)
    
    thresholds = [m['threshold'] for m in metrics]
    avg_ious = [m['avg_iou'] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, avg_ious, 'purple', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Average IoU', fontsize=12)
    plt.title('Segmentation Quality (IoU) vs Confidence', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0.0, 1.0])
    
    output_path = os.path.join(output_dir, 'iou_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def save_metrics_summary(metrics, optimal_threshold, output_dir='evaluation_results'):
    """Save metrics summary to text file"""
    os.makedirs(output_dir, exist_ok=True)
    
    optimal_metrics = next((m for m in metrics if abs(m['threshold'] - optimal_threshold) < 0.01), None)
    
    output_path = os.path.join(output_dir, 'metrics_summary.txt')
    
    with open(output_path, 'w') as f:
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        if optimal_metrics:
            f.write(f"Optimal Confidence Threshold: {optimal_threshold:.2f}\n\n")
            f.write("Metrics at Optimal Threshold:\n")
            f.write(f"  Precision: {optimal_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {optimal_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {optimal_metrics['f1']:.4f}\n")
            f.write(f"  Average IoU: {optimal_metrics['avg_iou']:.4f}\n\n")
            f.write("Detection Counts:\n")
            f.write(f"  True Positives: {optimal_metrics['tp']}\n")
            f.write(f"  False Positives: {optimal_metrics['fp']}\n")
            f.write(f"  False Negatives: {optimal_metrics['fn']}\n\n")
        
        f.write("Metrics Across All Thresholds:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}\n")
        f.write("-"*50 + "\n")
        
        for m in metrics:
            f.write(f"{m['threshold']:<12.2f} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                   f"{m['f1']:<12.4f} {m['avg_iou']:<12.4f}\n")
    
    print(f"✓ Saved: {output_path}")


def main():
    """Main execution function"""
    if len(sys.argv) < 3:
        print("Usage: python model_evaluation.py <test_images_directory> <model_id>")
        print("\nExamples:")
        print("  python model_evaluation.py ./test mastersthesis-d11wq/li6800-image-correction/11")
        print("\nFind your model ID at: Roboflow.com → Your Project → Deploy tab")
        print("Format: workspace/project-name/version")
        sys.exit(1)
    
    test_images_dir = sys.argv[1]
    model_id = sys.argv[2]
    
    if not os.path.isdir(test_images_dir):
        print(f"Error: Directory not found: {test_images_dir}")
        sys.exit(1)
    
    try:
        print("="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Setup LOCAL client
        client = setup_client()
        
        # Test model connection
        print(f"\nTesting model: {model_id}")
        test_files = list(Path(test_images_dir).glob('*.jpg'))[:1]
        if test_files:
            print(f"Running test inference on: {test_files[0].name}")
            try:
                test_result = run_inference_direct(client, str(test_files[0]), model_id)
                print(f"✓ Model is accessible and returning predictions")
                print(f"  Sample: {len(test_result.get('predictions', []))} predictions")
            except Exception as e:
                print(f"✗ Model test failed: {e}")
                print("\nPlease verify:")
                print("  1. Model ID is correct (workspace/project/version)")
                print("  2. Local inference server is running")
                print("  3. Model version exists in Roboflow")
                sys.exit(1)
        
        # Load ground truth
        print("\nLoading ground truth annotations...")
        annotations_path = "ground_truth/annotations.json"
        
        if not os.path.exists(annotations_path):
            print(f"\nError: {annotations_path} not found!")
            print("Run: py -3.11 download_annotations.py")
            sys.exit(1)
        
        with open(annotations_path, 'r') as f:
            ground_truth_data = json.load(f)
        
        print(f"Loaded {len(ground_truth_data)} annotated images")
        
        # Evaluate
        results = evaluate_model(client, test_images_dir, ground_truth_data, model_id)
        metrics = results['metrics_by_threshold']
        
        # Generate plots
        print("\nGenerating evaluation plots...")
        plot_precision_recall_curve(metrics)
        optimal_threshold, best_f1 = plot_f1_vs_threshold(metrics)
        plot_confusion_matrix(metrics, optimal_threshold)
        plot_iou_distribution(metrics)
        save_metrics_summary(metrics, optimal_threshold)
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETE")
        print("="*50)
        print(f"Optimal Threshold: {optimal_threshold:.2f}")
        print(f"Best F1 Score: {best_f1:.3f}")
        print("\nResults: evaluation_results/")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
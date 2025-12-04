#!/usr/bin/env python3
"""
Download Ground Truth Annotations from Roboflow
Converts Roboflow format to evaluation-ready format
"""

import sys
import os
import json
from pathlib import Path


def download_roboflow_dataset(api_key, workspace, project, version, output_dir='ground_truth'):
    """
    Download dataset from Roboflow using the official Python library
    
    Args:
        api_key: Your Roboflow API key
        workspace: Workspace ID (e.g., 'mastersthesis-d11wq')
        project: Project ID
        version: Dataset version number
        output_dir: Directory to save annotations
    
    Returns:
        Path to saved annotations file
    """
    print("="*50)
    print("DOWNLOADING ROBOFLOW ANNOTATIONS")
    print("="*50)
    
    try:
        from roboflow import Roboflow
    except ImportError:
        print("\nError: roboflow library not installed!")
        print("Please install it with: pip install roboflow")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset using Roboflow library
    print(f"\nDownloading dataset: {workspace}/{project}/{version}")
    print("This may take a moment...")
    
    try:
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("coco", location="./roboflow_download")
        
        print("Download successful!")
        
        # Find the annotations file
        annotation_files = list(Path("./roboflow_download").rglob("_annotations.coco.json"))
        
        if not annotation_files:
            raise FileNotFoundError("Could not find _annotations.coco.json in downloaded files")
        
        # Use the first annotations file found (usually from test or valid set)
        annotation_file = annotation_files[0]
        print(f"\nFound annotations: {annotation_file}")
        
        # Load and convert
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        print("\nConverting annotations to evaluation format...")
        ground_truth = convert_coco_to_eval_format(coco_data)
        
        # Save annotations
        output_file = os.path.join(output_dir, 'annotations.json')
        with open(output_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"Annotations saved to: {output_file}")
        print(f"Total images with annotations: {len(ground_truth)}")
        
        # Also save info about where images are located
        image_dir = annotation_file.parent
        print(f"Test images are located in: {image_dir}")
        
        return output_file, str(image_dir)
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nIf you're getting a 202 error, the dataset may still be generating.")
        print("Please try one of these alternatives:")
        print("  1. Wait 1-2 minutes and try again")
        print("  2. Download manually from Roboflow UI and use Option 2")
        raise


def convert_coco_to_eval_format(coco_data):
    """
    Convert COCO format annotations to evaluation format
    
    Args:
        coco_data: COCO format dataset
    
    Returns:
        Dictionary mapping image filenames to annotations
    """
    ground_truth = {}
    
    # Create mappings
    image_id_to_filename = {}
    for img in coco_data.get('images', []):
        image_id_to_filename[img['id']] = img['file_name']
    
    category_id_to_name = {}
    for cat in coco_data.get('categories', []):
        category_id_to_name[cat['id']] = cat['name']
    
    # Process annotations
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        filename = image_id_to_filename.get(image_id, 'unknown')
        
        if filename not in ground_truth:
            ground_truth[filename] = []
        
        # Convert segmentation polygon
        if 'segmentation' in ann and len(ann['segmentation']) > 0:
            # COCO format: [[x1, y1, x2, y2, ...]]
            seg = ann['segmentation'][0]
            points = [{'x': seg[i], 'y': seg[i+1]} for i in range(0, len(seg), 2)]
        else:
            # Use bounding box if no segmentation
            bbox = ann.get('bbox', [0, 0, 0, 0])
            x, y, w, h = bbox
            points = [
                {'x': x, 'y': y},
                {'x': x + w, 'y': y},
                {'x': x + w, 'y': y + h},
                {'x': x, 'y': y + h}
            ]
        
        annotation = {
            'class': category_id_to_name.get(ann['category_id'], 'unknown'),
            'points': points,
            'bbox': ann.get('bbox', []),
            'area': ann.get('area', 0),
            'annotation_id': ann['id']
        }
        
        ground_truth[filename].append(annotation)
    
    return ground_truth


def download_from_roboflow_ui_export(zip_path, output_dir='ground_truth'):
    """
    Process annotations from a Roboflow UI export (downloaded ZIP)
    
    Args:
        zip_path: Path to downloaded ZIP file from Roboflow
        output_dir: Directory to extract and process annotations
    
    Returns:
        Path to processed annotations file
    """
    import zipfile
    
    print("="*50)
    print("PROCESSING ROBOFLOW ZIP EXPORT")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    extract_dir = os.path.join(output_dir, 'extracted')
    
    # Extract ZIP
    print(f"\nExtracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print("Extraction complete!")
    
    # Look for _annotations.coco.json file
    annotation_files = list(Path(extract_dir).rglob('_annotations.coco.json'))
    
    if not annotation_files:
        raise FileNotFoundError("Could not find _annotations.coco.json in the extracted files")
    
    annotation_file = annotation_files[0]
    print(f"Found annotations: {annotation_file}")
    
    # Load and convert
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    print("\nConverting annotations...")
    ground_truth = convert_coco_to_eval_format(coco_data)
    
    # Save processed annotations
    output_file = os.path.join(output_dir, 'annotations.json')
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Annotations saved to: {output_file}")
    print(f"Total images with annotations: {len(ground_truth)}")
    
    # Find image directory
    image_dir = annotation_file.parent
    print(f"Test images are located in: {image_dir}")
    
    return output_file, str(image_dir)


def main():
    """Main execution function"""
    print("\nROBOFLOW ANNOTATION DOWNLOADER")
    print("="*50)
    print("\nOptions:")
    print("  1. Download via API (requires API key and roboflow library)")
    print("  2. Process downloaded ZIP file from Roboflow UI")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "1":
        print("\nAPI Download Selected")
        print("-"*50)
        
        api_key = input("Enter your Roboflow API key: ").strip()
        workspace = input("Enter workspace ID (default: mastersthesis-d11wq): ").strip() or "mastersthesis-d11wq"
        project = input("Enter project ID (default: li6800_image_correction-qkt8g): ").strip() or "li6800_image_correction-qkt8g"
        version = input("Enter dataset version (default: 9): ").strip() or "9"
        
        if not all([api_key, project, version]):
            print("Error: All fields are required!")
            sys.exit(1)
        
        try:
            output_file, image_dir = download_roboflow_dataset(api_key, workspace, project, int(version))
            print("\n" + "="*50)
            print("SUCCESS!")
            print("="*50)
            print(f"\nAnnotations ready at: {output_file}")
            print(f"\nTest images are in: {image_dir}")
            print(f"\nNext step: Run the evaluation script:")
            print(f'  py -3.11 model_evaluation.py "{image_dir}"')
            
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)
    
    elif choice == "2":
        print("\nZIP File Processing Selected")
        print("-"*50)
        
        zip_path = input("Enter path to downloaded ZIP file: ").strip()
        
        if not os.path.exists(zip_path):
            print(f"Error: File not found: {zip_path}")
            sys.exit(1)
        
        try:
            output_file, image_dir = download_from_roboflow_ui_export(zip_path)
            print("\n" + "="*50)
            print("SUCCESS!")
            print("="*50)
            print(f"\nAnnotations ready at: {output_file}")
            print(f"\nTest images are in: {image_dir}")
            print(f"\nNext step: Run the evaluation script:")
            print(f'  py -3.11 model_evaluation.py "{image_dir}"')
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        print("Invalid option selected!")
        sys.exit(1)


if __name__ == "__main__":
    main()

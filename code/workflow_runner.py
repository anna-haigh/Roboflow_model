#!/usr/bin/env python3
"""
Workflow Runner Script
Runs an inference workflow on local images using InferenceHTTPClient
Supports single images and batch processing of folders
"""

import sys
import os
import csv
from pathlib import Path
from inference_sdk import InferenceHTTPClient


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}


def setup_client():
    """Initialize and return the InferenceHTTPClient"""
    client = InferenceHTTPClient(
        api_url="http://localhost:9001",
        api_key="g76oBLBIPwupyLnfl0S0"
    )
    return client


def extract_pixel_counts(result):
    """
    Extract instrument and needle pixel counts from workflow results
    
    Args:
        result: Workflow results dictionary
    
    Returns:
        Tuple of (instrument_pixels, needle_pixels)
    """
    instrument_pixels = 0
    needle_pixels = 0
    
    # Print the result structure for debugging (first time only)
    if not hasattr(extract_pixel_counts, 'printed_structure'):
        print("\n" + "="*50)
        print("DEBUG: Result structure (first detection only)")
        print("="*50)
        if isinstance(result, list) and len(result) > 0:
            first_item = result[0]
            if 'predictions' in first_item:
                preds = first_item['predictions']
                if 'predictions' in preds and len(preds['predictions']) > 0:
                    print(f"Sample detection: {preds['predictions'][0].get('class', 'N/A')}")
        print("="*50 + "\n")
        extract_pixel_counts.printed_structure = True
    
    try:
        # The result is a list with one item containing predictions
        if isinstance(result, list) and len(result) > 0:
            for item in result:
                if isinstance(item, dict) and 'predictions' in item:
                    pred_data = item['predictions']
                    
                    # predictions has a 'predictions' list inside it
                    if isinstance(pred_data, dict) and 'predictions' in pred_data:
                        for detection in pred_data['predictions']:
                            if 'class' in detection and 'points' in detection:
                                class_name = detection['class'].lower()
                                # Count the number of points (pixels) in the polygon
                                pixel_count = len(detection['points'])
                                
                                if class_name == 'instrument':
                                    instrument_pixels += pixel_count
                                elif class_name in ['needle', 'leaf']:  # Count both needle and leaf
                                    needle_pixels += pixel_count
    except Exception as e:
        print(f"Warning: Could not extract pixel counts: {e}")
        import traceback
        traceback.print_exc()
    
    return instrument_pixels, needle_pixels


def run_workflow(client, image_path):
    """
    Run the workflow on a single image
    
    Args:
        client: InferenceHTTPClient instance
        image_path: Path to the image file
    
    Returns:
        Dictionary with results and pixel counts
    """
    # Verify image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Processing image: {image_path}")
    
    # Run workflow
    result = client.run_workflow(
        workspace_name="mastersthesis-d11wq",
        workflow_id="detect-count-and-visualize-5",
        images={
            "image": image_path
        },
        use_cache=True
    )
    
    # Extract pixel counts
    instrument_pixels, needle_pixels = extract_pixel_counts(result)
    
    # Calculate area ratio and corrected area
    total_pixels = needle_pixels + instrument_pixels
    if total_pixels > 0:
        area_ratio = needle_pixels / total_pixels
        corrected_area = area_ratio * 6
    else:
        area_ratio = 0
        corrected_area = 0
    
    return {
        'result': result,
        'instrument_pixels': instrument_pixels,
        'needle_pixels': needle_pixels,
        'area_ratio': area_ratio,
        'corrected_area': corrected_area
    }


def get_image_files(folder_path, recursive=False):
    """
    Get all image files from a directory
    
    Args:
        folder_path: Path to the folder
        recursive: Whether to search subdirectories
    
    Returns:
        List of image file paths
    """
    image_files = []
    path = Path(folder_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if recursive:
        pattern = '**/*'
    else:
        pattern = '*'
    
    for file_path in path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(str(file_path))
    
    return sorted(image_files)


def process_folder(client, folder_path, recursive=False, output_file="workflow_results.csv"):
    """
    Process all images in a folder
    
    Args:
        client: InferenceHTTPClient instance
        folder_path: Path to the folder
        recursive: Whether to search subdirectories
        output_file: Name of the output CSV file
    
    Returns:
        List of results
    """
    # Get all image files
    image_files = get_image_files(folder_path, recursive)
    
    if not image_files:
        print(f"No image files found in: {folder_path}")
        return []
    
    print("\n" + "="*50)
    print("BATCH PROCESSING")
    print("="*50)
    print(f"Found {len(image_files)} image(s)")
    print("="*50 + "\n")
    
    # Process each image
    all_results = []
    successful = 0
    failed = 0
    
    # Get the base folder path for subfolder identification
    base_folder = Path(folder_path).resolve()
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] ", end="")
        
        # Determine subfolder name
        image_path_obj = Path(image_path).resolve()
        try:
            # Get the relative path from base folder to image
            relative_path = image_path_obj.relative_to(base_folder)
            # Get the first parent directory (subfolder name)
            if len(relative_path.parts) > 1:
                subfolder_name = relative_path.parts[0]
            else:
                subfolder_name = "root"
        except ValueError:
            subfolder_name = "unknown"
        
        try:
            result_data = run_workflow(client, image_path)
            all_results.append({
                'file': image_path,
                'subfolder': subfolder_name,
                'status': 'success',
                'instrument_pixels': result_data['instrument_pixels'],
                'needle_pixels': result_data['needle_pixels'],
                'area_ratio': result_data['area_ratio'],
                'corrected_area': result_data['corrected_area']
            })
            successful += 1
        except Exception as e:
            print(f"FAILED - {e}")
            all_results.append({
                'file': image_path,
                'subfolder': subfolder_name,
                'status': 'failed',
                'instrument_pixels': None,
                'needle_pixels': None,
                'area_ratio': None,
                'corrected_area': None,
                'error': str(e)
            })
            failed += 1
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("="*50)
    
    # Save results to CSV
    if successful > 0:
        save_results_to_csv(all_results, output_file)
    
    return all_results


def save_results_to_csv(results, output_file):
    """
    Save results to a CSV file
    
    Args:
        results: List of result dictionaries
        output_file: Output CSV filename
    """
    print(f"\nSaving results to: {output_file}")
    
    # Try to open file, add number if it's locked
    attempt = 0
    while attempt < 10:
        try:
            if attempt > 0:
                base_name = output_file.rsplit('.', 1)[0]
                output_file = f"{base_name}_{attempt}.csv"
                print(f"File locked, trying: {output_file}")
            
            with open(output_file, 'w', newline='') as csvfile:
                fieldnames = ['subfolder', 'image_name', 'image_path', 'status', 'instrument_pixels', 
                             'needle_pixels', 'area_ratio', 'corrected_area']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in results:
                    # Extract just the filename from the full path
                    image_name = os.path.basename(result['file'])
                    
                    writer.writerow({
                        'subfolder': result.get('subfolder', ''),
                        'image_name': image_name,
                        'image_path': result['file'],
                        'status': result['status'],
                        'instrument_pixels': result.get('instrument_pixels', ''),
                        'needle_pixels': result.get('needle_pixels', ''),
                        'area_ratio': result.get('area_ratio', ''),
                        'corrected_area': result.get('corrected_area', '')
                    })
            
            print(f"Results saved successfully to: {output_file}")
            print("\nColumn descriptions:")
            print("  - subfolder: Name of the subfolder containing the image")
            print("  - image_name: Name of the image file")
            print("  - image_path: Full path to the image file")
            print("  - instrument_pixels: Total pixels detected as 'instrument'")
            print("  - needle_pixels: Total pixels detected as 'needle' or 'leaf'")
            print("  - area_ratio: needle_pixels / (needle_pixels + instrument_pixels)")
            print("  - corrected_area: area_ratio * 6")
            break
            
        except PermissionError:
            attempt += 1
            if attempt >= 10:
                print(f"\nError: Could not save CSV. Please close {output_file} if it's open in another program.")
                raise


def main():
    """Main execution function"""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python workflow_runner.py <image_path>")
        print("  Folder:        python workflow_runner.py <folder_path>")
        print("  Folder (recursive): python workflow_runner.py <folder_path> --recursive")
        print("\nExamples:")
        print("  python workflow_runner.py my_image.jpg")
        print("  python workflow_runner.py ./images/")
        print("  python workflow_runner.py ./images/ --recursive")
        sys.exit(1)
    
    path = sys.argv[1]
    recursive = '--recursive' in sys.argv
    
    try:
        # Setup client
        print("Connecting to inference server...")
        client = setup_client()
        
        # Check if path is a directory or file
        if os.path.isdir(path):
            # Process folder
            process_folder(client, path, recursive=recursive)
            
        elif os.path.isfile(path):
            # Process single image
            result_data = run_workflow(client, path)
            
            # Display results
            print("\n" + "="*50)
            print("WORKFLOW RESULTS")
            print("="*50)
            print(f"Instrument pixels: {result_data['instrument_pixels']}")
            print(f"Needle pixels: {result_data['needle_pixels']}")
            print(f"Area ratio: {result_data['area_ratio']:.4f}")
            print(f"Corrected area: {result_data['corrected_area']:.4f}")
            print("="*50)
            
        else:
            raise FileNotFoundError(f"Path not found: {path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running workflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

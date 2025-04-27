import os
import sys
import cv2
import numpy as np
import json
from collections import Counter
import re
import time
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from PIL import Image

# Import the OCR functions from Program.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Program import (
    ocr_tesseract,
    ocr_easyocr,
    ocr_paddleocr,
    ocr_trocr,
    ocr_yolo_crnn,
    ocr_donut,
    number_plate_detection
)

# Define model names for consistency
MODEL_NAMES = {
    'tesseract': 'Tesseract',
    'easyocr': 'EasyOCR',
    'paddleocr': 'PaddleOCR',
    'trocr': 'TrOCR',
    'yolo_crnn': 'YOLO+CRNN',
    'donut': 'Donut'
}

# Function to normalize text for comparison
def normalize_text(text):
    if not text:
        return ""
    # Remove spaces and special characters, convert to uppercase
    return str("".join(re.split("[^a-zA-Z0-9]*", text))).upper()

# Function to calculate text similarity score (0-100%)
def similarity_score(text1, text2):
    if not text1 and not text2:
        return 100.0
    if not text1 or not text2:
        return 0.0
        
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    return SequenceMatcher(None, text1, text2).ratio() * 100

# Function to preprocess the image for better OCR
def preprocess_image(img):
    # Resize image to a consistent size
    img = cv2.resize(img, (640, 480))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive threshold to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    return img, thresh

# Function to evaluate a single model
def evaluate_model(model_func, model_name, images, ground_truth):
    print(f"\nEvaluating {model_name}...")
    results = []
    start_time = time.time()
    
    # Used for calculating average metrics
    total_accuracy = 0
    total_similarity = 0
    total_confidence = 0
    correct_count = 0
    processed_count = 0
    
    for i, (img_path, truth) in enumerate(ground_truth.items()):
        if not os.path.exists(img_path):
            print(f"Warning: Image not found - {img_path}")
            continue
            
        img = images.get(img_path)
        if img is None:
            print(f"Warning: Failed to load image - {img_path}")
            continue
            
        # Original and preprocessed versions
        _, preprocessed = preprocess_image(img)
        
        # Try with both original and preprocessed image
        result_orig = model_func(img)
        result_prep = model_func(preprocessed)
        
        # Use the result with higher confidence
        result = result_orig if result_orig['confidence'] >= result_prep['confidence'] else result_prep
        
        # Normalize detected text
        detected_text = normalize_text(result['text'])
        truth_text = normalize_text(truth)
        
        # Calculate metrics
        is_correct = (detected_text == truth_text)
        sim_score = similarity_score(detected_text, truth_text)
        
        if is_correct:
            correct_count += 1
        
        total_similarity += sim_score
        total_confidence += result['confidence']
        processed_count += 1
        
        # Store individual result
        results.append({
            'image': os.path.basename(img_path),
            'ground_truth': truth_text,
            'detected': detected_text,
            'confidence': result['confidence'],
            'is_correct': is_correct,
            'similarity': sim_score
        })
        
        print(f"  Image {i+1}/{len(ground_truth)}: {'✓' if is_correct else '✗'} "
              f"GT: '{truth_text}', Detected: '{detected_text}', "
              f"Confidence: {result['confidence']:.2f}, Similarity: {sim_score:.1f}%")
    
    # Calculate averages
    accuracy = (correct_count / processed_count * 100) if processed_count > 0 else 0
    avg_similarity = (total_similarity / processed_count) if processed_count > 0 else 0
    avg_confidence = (total_confidence / processed_count) if processed_count > 0 else 0
    execution_time = time.time() - start_time
    
    summary = {
        'model': model_name,
        'accuracy': accuracy,
        'avg_similarity': avg_similarity,
        'avg_confidence': avg_confidence,
        'correct_count': correct_count,
        'total_count': processed_count,
        'execution_time': execution_time,
        'results': results
    }
    
    print(f"\n{model_name} Summary:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Average Similarity: {avg_similarity:.2f}%")
    print(f"  Average Confidence: {avg_confidence:.2f}")
    print(f"  Execution Time: {execution_time:.2f} seconds")
    
    return summary

# Function to evaluate the combined detection + OCR pipeline
def evaluate_full_pipeline(images, ground_truth):
    print("\nEvaluating Full Detection Pipeline...")
    results = []
    start_time = time.time()
    
    total_similarity = 0
    correct_count = 0
    processed_count = 0
    
    for i, (img_path, truth) in enumerate(ground_truth.items()):
        if not os.path.exists(img_path):
            print(f"Warning: Image not found - {img_path}")
            continue
            
        img = images.get(img_path)
        if img is None:
            print(f"Warning: Failed to load image - {img_path}")
            continue
        
        # Run the full detection pipeline
        detected = number_plate_detection(img)
        
        # Normalize texts
        detected_text = normalize_text(detected)
        truth_text = normalize_text(truth)
        
        # Calculate metrics
        is_correct = (detected_text == truth_text)
        sim_score = similarity_score(detected_text, truth_text)
        
        if is_correct:
            correct_count += 1
        
        total_similarity += sim_score
        processed_count += 1
        
        # Store individual result
        results.append({
            'image': os.path.basename(img_path),
            'ground_truth': truth_text,
            'detected': detected_text,
            'is_correct': is_correct,
            'similarity': sim_score
        })
        
        print(f"  Image {i+1}/{len(ground_truth)}: {'✓' if is_correct else '✗'} "
              f"GT: '{truth_text}', Detected: '{detected_text}', "
              f"Similarity: {sim_score:.1f}%")
    
    # Calculate averages
    accuracy = (correct_count / processed_count * 100) if processed_count > 0 else 0
    avg_similarity = (total_similarity / processed_count) if processed_count > 0 else 0
    execution_time = time.time() - start_time
    
    summary = {
        'model': 'Full Pipeline',
        'accuracy': accuracy,
        'avg_similarity': avg_similarity,
        'correct_count': correct_count,
        'total_count': processed_count,
        'execution_time': execution_time,
        'results': results
    }
    
    print(f"\nFull Pipeline Summary:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Average Similarity: {avg_similarity:.2f}%")
    print(f"  Execution Time: {execution_time:.2f} seconds")
    
    return summary

# Function to create a visualization of the results
def create_visualizations(evaluation_results, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    models = []
    accuracies = []
    similarities = []
    confidences = []
    execution_times = []
    
    for result in evaluation_results:
        if result['model'] != 'Full Pipeline':  # Only include individual models
            models.append(result['model'])
            accuracies.append(result['accuracy'])
            similarities.append(result['avg_similarity'])
            confidences.append(result.get('avg_confidence', 0))
            execution_times.append(result['execution_time'])
    
    # Create accuracy comparison chart
    plt.figure(figsize=(12, 6))
    plt.bar(models, accuracies, color='skyblue')
    plt.title('OCR Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    
    # Create similarity comparison chart
    plt.figure(figsize=(12, 6))
    plt.bar(models, similarities, color='lightgreen')
    plt.title('OCR Model Text Similarity Comparison')
    plt.ylabel('Average Similarity (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    for i, v in enumerate(similarities):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_comparison.png'))
    
    # Create confidence comparison chart
    plt.figure(figsize=(12, 6))
    plt.bar(models, confidences, color='coral')
    plt.title('OCR Model Confidence Comparison')
    plt.ylabel('Average Confidence')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    for i, v in enumerate(confidences):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_comparison.png'))
    
    # Create execution time comparison chart
    plt.figure(figsize=(12, 6))
    plt.bar(models, execution_times, color='orchid')
    plt.title('OCR Model Execution Time Comparison')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(rotation=45)
    for i, v in enumerate(execution_times):
        plt.text(i, v + 0.1, f"{v:.1f}s", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
    
    print(f"Visualizations saved to {output_dir}")

def main():
    # Define directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    np_dir = os.path.join(current_dir, "Number_Plate_Detection")
    dataset_dir = os.path.join(np_dir, "Dataset")
    results_dir = os.path.join(current_dir, "evaluation_results")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print("License Plate OCR Model Evaluation")
    print("==================================")
    
    # Ask for ground truth data
    print("\nTo evaluate model accuracy, we need ground truth data (actual license plate numbers).")
    print("Choose an option:")
    print("1. Enter ground truth data manually")
    print("2. Load from a JSON file")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    ground_truth = {}
    
    if choice == "1":
        print("\nEntering ground truth data manually.")
        print("For each image, enter the correct license plate number.")
        print("Enter 'skip' to skip an image or 'done' when finished.")
        
        # Get all images in the dataset directory
        image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(dataset_dir, img_file)
            
            # Display the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
                
            img_display = cv2.resize(img, (600, 400))
            cv2.imshow(f"Image: {img_file}", img_display)
            cv2.waitKey(500)  # Show for half a second
            
            # Get ground truth from user
            plate_number = input(f"\nEnter correct license plate for {img_file} (or 'skip'/'done'): ").strip()
            
            cv2.destroyAllWindows()
            
            if plate_number.lower() == 'done':
                break
            elif plate_number.lower() == 'skip':
                continue
            else:
                ground_truth[img_path] = plate_number
        
        # Save ground truth to a file for future use
        with open(os.path.join(results_dir, 'ground_truth.json'), 'w') as f:
            json.dump({os.path.basename(k): v for k, v in ground_truth.items()}, f, indent=2)
            
        print(f"\nGround truth data saved to {os.path.join(results_dir, 'ground_truth.json')}")
        
    elif choice == "2":
        gt_file = input("Enter path to ground truth JSON file (or press Enter for default): ").strip()
        
        if not gt_file:
            gt_file = os.path.join(results_dir, 'ground_truth.json')
        
        try:
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)
                
            # Convert relative image paths to absolute paths
            for img_name, plate in gt_data.items():
                img_path = os.path.join(dataset_dir, img_name)
                ground_truth[img_path] = plate
                
            print(f"Loaded {len(ground_truth)} ground truth entries.")
            
        except Exception as e:
            print(f"Error loading ground truth file: {e}")
            return
    else:
        print("Invalid choice. Please run the program again.")
        return
    
    # Load all images into memory
    print("\nLoading images...")
    images = {}
    for img_path in ground_truth.keys():
        if os.path.exists(img_path):
            images[img_path] = cv2.imread(img_path)
    
    print(f"Loaded {len(images)} images for evaluation.\n")
    
    # Define models to evaluate
    models = {
        'tesseract': (ocr_tesseract, MODEL_NAMES['tesseract']),
        'easyocr': (ocr_easyocr, MODEL_NAMES['easyocr']),
        'paddleocr': (ocr_paddleocr, MODEL_NAMES['paddleocr']),
        'trocr': (ocr_trocr, MODEL_NAMES['trocr']),
        'yolo_crnn': (ocr_yolo_crnn, MODEL_NAMES['yolo_crnn']),
        'donut': (ocr_donut, MODEL_NAMES['donut'])
    }
    
    # Allow user to select which models to evaluate
    print("Select models to evaluate (comma-separated list or 'all'):")
    for i, (key, (_, name)) in enumerate(models.items(), 1):
        print(f"{i}. {name}")
    print(f"{len(models) + 1}. Full Pipeline (license plate detection + OCR)")
    
    model_choice = input("\nEnter your choices: ").strip().lower()
    
    evaluation_results = []
    
    if model_choice == 'all':
        # Evaluate all individual models
        for model_key, (model_func, model_name) in models.items():
            result = evaluate_model(model_func, model_name, images, ground_truth)
            evaluation_results.append(result)
            
        # Evaluate the full pipeline
        pipeline_result = evaluate_full_pipeline(images, ground_truth)
        evaluation_results.append(pipeline_result)
        
    else:
        try:
            choices = [int(x.strip()) for x in model_choice.split(',')]
            
            for choice in choices:
                if 1 <= choice <= len(models):
                    # Evaluate individual model
                    model_key = list(models.keys())[choice - 1]
                    model_func, model_name = models[model_key]
                    result = evaluate_model(model_func, model_name, images, ground_truth)
                    evaluation_results.append(result)
                elif choice == len(models) + 1:
                    # Evaluate full pipeline
                    pipeline_result = evaluate_full_pipeline(images, ground_truth)
                    evaluation_results.append(pipeline_result)
                else:
                    print(f"Invalid choice: {choice}")
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers.")
            return
    
    # Save evaluation results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'evaluation_results_{timestamp}.json')
    
    # Convert results to serializable format
    serializable_results = []
    for result in evaluation_results:
        serializable_result = result.copy()
        serializable_results.append(serializable_result)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")
    
    # Create visualizations
    create_visualizations(evaluation_results, results_dir)
    
    # Print overall comparison
    print("\nModel Accuracy Comparison:")
    print("-------------------------")
    sorted_results = sorted(evaluation_results, key=lambda x: x['accuracy'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['model']}: {result['accuracy']:.2f}% accuracy, " +
              f"{result['avg_similarity']:.2f}% similarity")

if __name__ == "__main__":
    main()
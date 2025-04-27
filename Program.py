import sys
import glob
import os
import numpy as np
import cv2
from PIL import Image
import pytesseract
import re
import easyocr
from paddleocr import PaddleOCR
import argparse
from collections import Counter
import json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoFeatureExtractor
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure pytesseract with the path to the tesseract executable in the OCR folder
current_dir = os.path.dirname(__file__)
pytesseract.pytesseract.tesseract_cmd = os.path.join(current_dir, 'OCR', 'tesseract.exe')

# Update paths to use trained models
trained_models_dir = os.path.join(current_dir, 'trained_models')

# Load custom Tesseract config if available
tesseract_config_path = os.path.join(trained_models_dir, 'license_plate.config')
tesseract_custom_config = '--psm 7 --oem 3'
if os.path.exists(tesseract_config_path):
    with open(tesseract_config_path, 'r') as f:
        tesseract_custom_config = f.read().replace('\n', ' ')
    print(f"Using custom Tesseract config: {tesseract_custom_config}")

# Check if TrOCR was trained
trocr_config_path = os.path.join(trained_models_dir, 'trocr_config.json')
use_trained_trocr = os.path.exists(trocr_config_path)
if use_trained_trocr:
    print("Using trained TrOCR model for license plate recognition")

# Global variables for model instances (lazy loading)
easyocr_reader = None
paddleocr_model = None
trocr_processor = None
trocr_model = None
yolo_model = None
crnn_model = None
donut_processor = None
donut_model = None

# Update Tesseract wrapper to use custom config
def ocr_tesseract(image):
    try:
        text = pytesseract.image_to_string(image, lang='eng', config=tesseract_custom_config)
        return {'text': text.strip(), 'confidence': 0.7}  # Estimated confidence
    except Exception as e:
        print(f"Tesseract OCR Error: {e}")
        return {'text': '', 'confidence': 0}

# Wrapper function for EasyOCR
def ocr_easyocr(image):
    try:
        global easyocr_reader
        if easyocr_reader is None:
            print("Initializing EasyOCR model...")
            easyocr_reader = easyocr.Reader(['en'])
        
        results = easyocr_reader.readtext(np.array(image))
        if results:
            text, confidence = results[0][1], results[0][2]
            return {'text': text, 'confidence': confidence}
        return {'text': '', 'confidence': 0}
    except Exception as e:
        print(f"EasyOCR Error: {e}")
        return {'text': '', 'confidence': 0}

# Wrapper function for PaddleOCR
def ocr_paddleocr(image):
    try:
        global paddleocr_model
        if paddleocr_model is None:
            print("Initializing PaddleOCR model...")
            paddleocr_model = PaddleOCR(lang='en')
        
        results = paddleocr_model.ocr(np.array(image), cls=True)
        if results and len(results[0]) > 0:
            text, confidence = results[0][0][1][0], results[0][0][1][1]
            return {'text': text, 'confidence': confidence}
        return {'text': '', 'confidence': 0}
    except Exception as e:
        print(f"PaddleOCR Error: {e}")
        return {'text': '', 'confidence': 0}

# Wrapper function for TrOCR (Microsoft's Transformer OCR)
def ocr_trocr(image):
    try:
        global trocr_processor, trocr_model
        if trocr_processor is None or trocr_model is None:
            print("Initializing TrOCR model... (downloading on first use)")
            try:
                # Suppress warnings about uninitialized weights - this is expected for inference
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Some weights of VisionEncoderDecoderModel")
                    warnings.filterwarnings("ignore", message="You should probably TRAIN this model")
                    
                    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
                    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed", ignore_mismatched_sizes=True)
                    
                if torch.cuda.is_available():
                    trocr_model.to("cuda")
            except Exception as e:
                print(f"Failed to load TrOCR model: {e}")
                return {'text': '', 'confidence': 0}
        
        # Convert image to RGB if it's grayscale
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 1:  # Single channel
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Convert to PIL Image
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
            
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
        if torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")
            
        generated_ids = trocr_model.generate(pixel_values)
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # TrOCR doesn't provide confidence scores, using a fixed value
        return {'text': generated_text, 'confidence': 0.8}
    except Exception as e:
        print(f"TrOCR Error: {e}")
        return {'text': '', 'confidence': 0}

# Wrapper function for YOLO+CRNN Combo
def ocr_yolo_crnn(image):
    try:
        # Since full YOLO+CRNN implementation is complex,
        # this is a placeholder that would use a pre-trained YOLO for text detection
        # and CRNN for recognition
        global yolo_model, crnn_model
        
        # For now, return estimated results
        # In a real implementation, this would detect text regions with YOLO
        # and then recognize each region with CRNN
        
        # Fallback to Tesseract for now
        tesseract_result = ocr_tesseract(image)
        return {'text': tesseract_result['text'], 'confidence': 0.6}  # Lower confidence since this is a fallback
    except Exception as e:
        print(f"YOLO+CRNN Error: {e}")
        return {'text': '', 'confidence': 0}

# Wrapper function for Donut model
def ocr_donut(image):
    try:
        global donut_processor, donut_model
        
        # Placeholder for Donut implementation
        # In a real implementation, we would load and use the Donut model
        
        # Fallback to Tesseract for now
        tesseract_result = ocr_tesseract(image)
        return {'text': tesseract_result['text'], 'confidence': 0.5}  # Lower confidence since this is a fallback
    except Exception as e:
        print(f"Donut Error: {e}")
        return {'text': '', 'confidence': 0}

# Function to run all OCR models on an image
def run_all_ocr_models(image, selected_models=None):
    results = {}
    models_to_run = {}
    
    # Define all available models
    all_models = {
        'tesseract': ocr_tesseract,
        'easyocr': ocr_easyocr,
        'paddleocr': ocr_paddleocr,
        'trocr': ocr_trocr,
        'yolo_crnn': ocr_yolo_crnn,
        'donut': ocr_donut
    }
    
    # If selected_models is provided, use only those models
    if selected_models:
        for model in selected_models:
            if model in all_models:
                models_to_run[model] = all_models[model]
            else:
                print(f"Warning: Model '{model}' not found")
    else:
        # Default to using all models
        models_to_run = all_models
    
    # Run each selected model
    for model_name, model_func in models_to_run.items():
        print(f"Running {model_name}...")
        result = model_func(image)
        # Print just the detected plate text for each model
        if result['text']:
            print(f"{model_name}: {result['text']}")
        results[model_name] = result
    
    return results

# Function to select the best OCR result using voting
def select_best_ocr_result(results):
    # Extract texts and their confidences
    texts = []
    confidence_map = {}
    
    for model_name, result in results.items():
        text = result['text']
        if text:
            # Normalize text: remove spaces, uppercase
            normalized_text = ''.join(text.upper().split())
            confidence = result['confidence']
            
            texts.append(normalized_text)
            
            # Store highest confidence for this text
            if normalized_text not in confidence_map or confidence > confidence_map[normalized_text]['confidence']:
                confidence_map[normalized_text] = {
                    'confidence': confidence,
                    'model': model_name
                }
    
    # Count occurrences of each text
    text_counts = Counter(texts)
    
    # If we have results
    if text_counts:
        # Get the most common text(s)
        most_common_texts = text_counts.most_common()
        
        # If there's a clear winner by vote count
        if len(most_common_texts) == 1 or (len(most_common_texts) > 1 and most_common_texts[0][1] > most_common_texts[1][1]):
            winner_text = most_common_texts[0][0]
            winner_info = confidence_map[winner_text]
            return {
                'text': winner_text,
                'confidence': winner_info['confidence'],
                'model': winner_info['model'],
                'vote_count': most_common_texts[0][1],
                'total_votes': len(texts)
            }
        else:
            # If tie, choose the one with highest confidence
            tied_texts = [t for t, c in most_common_texts if c == most_common_texts[0][1]]
            best_text = max(tied_texts, key=lambda t: confidence_map[t]['confidence'])
            best_info = confidence_map[best_text]
            
            return {
                'text': best_text,
                'confidence': best_info['confidence'],
                'model': best_info['model'],
                'vote_count': text_counts[best_text],
                'total_votes': len(texts)
            }
    
    # No results
    return {
        'text': '',
        'confidence': 0,
        'model': None,
        'vote_count': 0,
        'total_votes': 0
    }

def number_plate_detection(img, ocr_models=None):
    def clean2_plate(plate):
        gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    
        # Try multiple threshold values for better results
        thresholds = [90, 100, 110, 120, 130]
        best_text = ""
        
        for thresh_val in thresholds:
            _, thresh = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
            if cv2.waitKey(0) & 0xff == ord('q'):
                pass
                
            num_contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
            if num_contours:
                contour_area = [cv2.contourArea(c) for c in num_contours]
                max_cntr_index = np.argmax(contour_area)
        
                max_cnt = num_contours[max_cntr_index]
                max_cntArea = contour_area[max_cntr_index]
                x,y,w,h = cv2.boundingRect(max_cnt)
        
                if not ratioCheck(max_cntArea,w,h):
                    continue
        
                final_img = thresh[y:y+h, x:x+w]
                
                # Try OCR on this threshold version
                plate_im = Image.fromarray(final_img)
                try:
                    text = pytesseract.image_to_string(plate_im, lang='eng', config='--psm 7 --oem 3')
                    text = text.strip()
                    if text and len(text) > 2:  # Must have at least 3 characters
                        best_text = text
                        return final_img, [x,y,w,h], best_text
                except Exception as e:
                    print(f"OCR Error: {e}")
                    continue
        
        # If we found no good text but have contours, return the best threshold result
        if num_contours:
            _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
            contour_area = [cv2.contourArea(c) for c in num_contours]
            max_cntr_index = np.argmax(contour_area)
            max_cnt = num_contours[max_cntr_index]
            x,y,w,h = cv2.boundingRect(max_cnt)
            final_img = thresh[y:y+h, x:x+w]
            return final_img, [x,y,w,h], ""
            
        return plate, None, ""
    
    def ratioCheck(area, width, height):
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        # More flexible ratio constraints
        if (area < 800 or area > 85000) or (ratio < 2 or ratio > 8):
            return False
        return True
    
    def isMaxWhite(plate):
        avg = np.mean(plate)
        # More flexible white threshold
        if(avg >= 90):
            return True
        else:
            return False
    
    def ratio_and_rotation(rect):
        (x, y), (width, height), rect_angle = rect
    
        if(width>height):
            angle = -rect_angle
        else:
            angle = 90 + rect_angle
    
        # More flexible angle threshold
        if angle>30:
            return False
    
        if height == 0 or width == 0:
            return False
    
        area = height*width
        if not ratioCheck(area,width,height):
            return False
        else:
            return True
    
    def apply_ocr(plate_img):
        if ocr_models:
            all_results = run_all_ocr_models(plate_img, ocr_models)
            best_result = select_best_ocr_result(all_results)
            return best_result
        else:
            return ocr_tesseract(plate_img)

    # Try different preprocessing methods
    results = []
    
    # Original method
    img2 = cv2.GaussianBlur(img, (5,5), 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    img2 = cv2.Sobel(img2,cv2.CV_8U,1,0,ksize=3)    
    _,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = img2.copy()
    cv2.morphologyEx(src=img2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    num_contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img2, num_contours, -1, (0,255,0), 1)
    
    # Try all promising contours
    for i,cnt in enumerate(num_contours):
        min_rect = cv2.minAreaRect(cnt)
        if ratio_and_rotation(min_rect):
            x,y,w,h = cv2.boundingRect(cnt)
            plate_img = img[y:y+h,x:x+w]
            if(isMaxWhite(plate_img)):
                # Display candidate plate region for 500ms
                cv2.imshow("Candidate plate region", cv2.resize(plate_img, (200, 100)))
                cv2.waitKey(500)
                cv2.destroyWindow("Candidate plate region")
                
                clean_plate, rect, text = clean2_plate(plate_img)
                if rect and text:
                    ocr_output = apply_ocr(plate_img)
                    if ocr_output['text']:
                        results.append(ocr_output['text'])
    
    # If we didn't get any results with the first method, try direct OCR
    if not results:
        # Try alternative approach - direct OCR on preprocessed regions
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)
        
        edged = cv2.Canny(gray, 30, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            
            if len(approx) == 4:
                x,y,w,h = cv2.boundingRect(c)
                
                # Check if the region is reasonably sized
                if w > 100 and h > 30 and w < img.shape[1]*0.9:
                    plate_img = img[y:y+h, x:x+w]
                    
                    # Display candidate plate region
                    cv2.imshow("Candidate plate region", cv2.resize(plate_img, (200, 100)))
                    cv2.waitKey(500)
                    cv2.destroyWindow("Candidate plate region")
                    
                    # Run OCR models
                    all_results = run_all_ocr_models(plate_img, ocr_models)
                    best_result = select_best_ocr_result(all_results)
                    
                    if best_result['text']:
                        # Add the text result
                        results.append(best_result['text'])
    
    # Return the best result
    if results:
        # Clean and filter results
        for i, text in enumerate(results):
            results[i] = str("".join(re.split("[^a-zA-Z0-9]*", text)))
            
        # Choose the result with reasonable length (most likely a license plate)
        filtered = [r for r in results if 5 <= len(r) <= 12]
        if filtered:
            return max(filtered, key=len)
        elif results:
            return max(results, key=len)
    
    return ""

# Quick sort
def partition(arr,low,high): 
    i = ( low-1 )         
    pivot = arr[high]    
  
    for j in range(low , high): 
        if   arr[j] < pivot: 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 

def quickSort(arr,low,high): 
    if low < high: 
        pi = partition(arr,low,high) 
  
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high)
        
    return arr

print("HELLO!!")
print("Welcome to the Number Plate Detection System.\n")

# Allow user to choose between dataset or custom image
print("Choose an option:")
print("1. Process images from Dataset folder")
print("2. Process a custom image (provide the full path)")
print("3. Select specific OCR models to use")

choice = input("Enter your choice (1, 2, or 3): ")

array=[]

# Define available models
AVAILABLE_MODELS = ['tesseract', 'easyocr', 'paddleocr', 'trocr', 'yolo_crnn', 'donut']
# By default, use all models
selected_models = AVAILABLE_MODELS.copy()

if choice == "1":
    # Images are in the Search_Image folder
    search_img_dir = os.path.join(current_dir, "Search_Image")
    
    print(f"Looking for images in: {search_img_dir}")
    print("Displaying all images from the Search_Image folder...\n")
    
    # Look for images in the Search_Image folder (support multiple image formats)
    for img_path in glob.glob(os.path.join(search_img_dir, "*.[jp][pn]g")) + glob.glob(os.path.join(search_img_dir, "*.jpeg")):
        print(f"Processing image: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
            
        img2 = cv2.resize(img, (600, 600))
        cv2.imshow(f"Image: {os.path.basename(img_path)}", img2)
        
        # Show image for 2 seconds instead of waiting for key press
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        
        # Process the license plate with selected models
        number_plate = number_plate_detection(img, ocr_models=selected_models)
        if number_plate:
            res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate))).upper()
            print(f"Detected license plate: {res2}\n")
            if res2:  # Only add non-empty results
                array.append(res2)
        else:
            print(f"No license plate detected in {os.path.basename(img_path)}\n")
            
elif choice == "2":
    custom_img_path = input("Enter the full path to your image: ")
    
    # Check if the file exists first
    if os.path.exists(custom_img_path):
        img = cv2.imread(custom_img_path)
    else:
        # If the file doesn't exist, try to add an extension
        _, ext = os.path.splitext(custom_img_path)
        if not ext:
            # Try common image extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = custom_img_path + ext
                if os.path.exists(test_path):
                    custom_img_path = test_path
                    print(f"Using image file: {custom_img_path}")
                    img = cv2.imread(custom_img_path)
                    break
            else:
                img = None
                print(f"File not found: {custom_img_path}")
        else:
            img = None
            print(f"File not found: {custom_img_path}")
    
    # Process the image if we found it
    if img is not None:
        img2 = cv2.resize(img, (600, 600))
        cv2.imshow(f"Custom Image", img2)
        
        print("Processing custom image...")
        # Show image for 2 seconds instead of waiting for key press
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        
        # Process the license plate with selected models
        number_plate = number_plate_detection(img, ocr_models=selected_models)
        if number_plate:
            res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate))).upper()
            print(f"Detected license plate: {res2}\n")
            array.append(res2)
        else:
            print("No license plate detected in the custom image\n")

elif choice == "3":
    # Let the user select which OCR models to use
    print("\nSelect which OCR models to use (comma-separated numbers, e.g., 1,3,4):")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"{i}. {model.upper()}")
    
    model_choice = input("Enter your choices (or 'all' for all models): ").strip()
    
    if model_choice.lower() == 'all':
        selected_models = AVAILABLE_MODELS.copy()
    else:
        try:
            # Parse user input and select models
            selected_indices = [int(idx.strip()) for idx in model_choice.split(',')]
            selected_models = [AVAILABLE_MODELS[idx-1] for idx in selected_indices if 1 <= idx <= len(AVAILABLE_MODELS)]
            
            if not selected_models:
                print("No valid models selected. Using all models.")
                selected_models = AVAILABLE_MODELS.copy()
        except Exception as e:
            print(f"Error parsing model selection: {e}. Using all models.")
            selected_models = AVAILABLE_MODELS.copy()
    
    print(f"\nSelected models: {', '.join([m.upper() for m in selected_models])}\n")
    
    # Now ask which option to proceed with
    print("Choose what to process with selected models:")
    print("1. Process images from Dataset folder")
    print("2. Process a custom image")
    
    process_choice = input("Enter your choice (1 or 2): ")
    
    if process_choice == "1":
        dataset_dir = os.path.join(current_dir, "Dataset")
        
        print(f"Looking for images in: {dataset_dir}")
        print("Displaying all images from the Dataset folder...\n")
        
        for img_path in glob.glob(os.path.join(dataset_dir, "*.jpeg")):
            print(f"Processing image: {os.path.basename(img_path)}")
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
                
            img2 = cv2.resize(img, (600, 600))
            cv2.imshow(f"Image: {os.path.basename(img_path)}", img2)
            
            # Show image for 2 seconds instead of waiting for key press
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            
            # Process the license plate with selected models
            number_plate = number_plate_detection(img, ocr_models=selected_models)
            if number_plate:
                res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate))).upper()
                print(f"Detected license plate: {res2}\n")
                if res2:
                    array.append(res2)
            else:
                print(f"No license plate detected in {os.path.basename(img_path)}\n")
    
    elif process_choice == "2":
        custom_img_path = input("Enter the full path to your image: ")
        if os.path.exists(custom_img_path):
            img = cv2.imread(custom_img_path)
            
            if img is None:
                print(f"Could not read image: {custom_img_path}")
            else:
                img2 = cv2.resize(img, (600, 600))
                cv2.imshow(f"Custom Image", img2)
                
                print("Processing custom image...")
                # Show image for 2 seconds instead of waiting for key press
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                
                # Process the license plate with selected models
                number_plate = number_plate_detection(img, ocr_models=selected_models)
                if number_plate:
                    res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate))).upper()
                    print(f"Detected license plate: {res2}\n")
                    array.append(res2)
                else:
                    print("No license plate detected in the custom image\n")
        else:
            print(f"File not found: {custom_img_path}")
    else:
        print("Invalid processing choice.")
else:
    print("Invalid choice. Please run the program again and select option 1 or 2.")

# Make sure we don't have duplicate entries
if array:
    array = list(set(array))
    if len(array) > 1:  # Only sort if we have multiple entries
        array = quickSort(array, 0, len(array)-1)
    
    print("\n\nSummary of detected license plates:")
    for i in array:
        print(i)
else:
    print("No license plates were detected in any of the images.")

print("\nThank you for using the Number Plate Detection System!")
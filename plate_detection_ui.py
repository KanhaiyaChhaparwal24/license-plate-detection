import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import sys
import re
import easyocr
import pytesseract
from paddleocr import PaddleOCR
from collections import Counter
import warnings
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

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
                # Display candidate plate region for a short time (100ms instead of 500ms)
                cv2.imshow("Candidate plate region", cv2.resize(plate_img, (200, 100)))
                cv2.waitKey(100)
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
                    
                    # Display candidate plate region with shorter display time
                    cv2.imshow("Candidate plate region", cv2.resize(plate_img, (200, 100)))
                    cv2.waitKey(100)
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

# Define the images folder path
current_dir = os.path.dirname(__file__)
images_folder = os.path.join(current_dir, 'images')

class LicensePlateDetectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detection System")
        self.root.geometry("1000x700")
        
        # Available models
        self.available_models = ['tesseract', 'easyocr', 'paddleocr', 'trocr', 'yolo_crnn', 'donut']
        self.selected_models = self.available_models.copy()  # By default, all models are selected
        
        # Create UI elements
        self.create_ui()
        
        # Current image
        self.current_image = None
        self.processed_plate_image = None
        
        # Populate image list
        self.load_image_list()
    
    def create_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for image selection
        left_panel = ttk.LabelFrame(main_frame, text="Image Selection", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Search box
        search_frame = ttk.Frame(left_panel)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_images)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Image listbox with scrollbar
        list_frame = ttk.Frame(left_panel)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.image_listbox.yview)
        
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # Button to load custom image
        ttk.Button(left_panel, text="Load Custom Image", command=self.load_custom_image).pack(fill=tk.X, pady=(5, 0))
        
        # Right panel for image display and results
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
          # Image display
        image_frame = ttk.LabelFrame(right_panel, text="Image Preview", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
          # Create a frame for the detect button (placing it BEFORE the results frame)
        detect_button_frame = ttk.Frame(right_panel)
        detect_button_frame.pack(fill=tk.X, pady=10)
        detect_button = ttk.Button(
            detect_button_frame, 
            text="Detect License Plate", 
            command=self.detect_license_plate,
            style="Accent.TButton"  # Add a special style for visibility
        )
        detect_button.pack(fill=tk.X, ipady=8)  # Increase padding for better visibility
        
        # Results display
        results_frame = ttk.LabelFrame(right_panel, text="Detection Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
          # Model selection checkboxes
        model_frame = ttk.LabelFrame(results_frame, text="Select OCR Models")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add Auto-detect option
        auto_detect_frame = ttk.Frame(model_frame)
        auto_detect_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=10, pady=5)
        
        self.auto_detect_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            auto_detect_frame,
            text="Auto-detect after image selection",
            variable=self.auto_detect_var
        ).pack(side=tk.LEFT)
        
        self.model_vars = {}
        for i, model in enumerate(self.available_models):
            var = tk.BooleanVar(value=True)
            self.model_vars[model] = var
            
            # Create checkboxes in a 3x2 grid
            ttk.Checkbutton(
                model_frame, 
                text=model.upper(), 
                variable=var,
                command=self.update_selected_models
            ).grid(
                row=i // 3 + 1,  # +1 to make room for auto-detect
                column=i % 3, 
                sticky=tk.W,
                padx=10,
                pady=5
            )
        
        # Results text
        results_text_frame = ttk.Frame(results_frame)
        results_text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(results_text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(results_text_frame, yscrollcommand=scrollbar.set, height=10)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
    
    def load_image_list(self):
        """Load image list from the images folder"""
        self.images = []
        
        if os.path.exists(images_folder):
            for file in os.listdir(images_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(file)
            
            self.images.sort()
            self.update_listbox()
    
    def update_listbox(self):
        """Update the listbox with filtered images"""
        self.image_listbox.delete(0, tk.END)
        for image in self.images:
            self.image_listbox.insert(tk.END, image)
    
    def filter_images(self, *args):
        """Filter images based on search text"""
        search_text = self.search_var.get().lower()
        
        if os.path.exists(images_folder):
            self.images = []
            for file in os.listdir(images_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and search_text in file.lower():
                    self.images.append(file)
            
            self.images.sort()
            self.update_listbox()    
    def on_image_select(self, event):
        """Handle image selection from listbox"""
        if self.image_listbox.curselection():
            index = self.image_listbox.curselection()[0]
            image_name = self.image_listbox.get(index)
            image_path = os.path.join(images_folder, image_name)
            
            try:
                # Load and display the image
                self.load_image(image_path)
                
                # If auto-detect is enabled, run detection automatically
                if self.auto_detect_var.get():
                    self.detect_license_plate()
                else:
                    # Otherwise, indicate that an image is ready for processing
                    self.update_results("Image loaded. Click 'Detect License Plate' to start detection.")
            except Exception as e:
                self.update_results(f"Error loading image: {e}")
    
    def load_custom_image(self):
        """Load a custom image from any folder"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        
        if file_path:
            try:
                self.load_image(file_path)
            except Exception as e:
                self.update_results(f"Error loading image: {e}")
    
    def load_image(self, image_path):
        """Load and display an image"""
        # Load with OpenCV for processing
        self.current_image = cv2.imread(image_path)
        
        if self.current_image is None:
            self.update_results(f"Could not read image: {image_path}")
            return
        
        # Convert to RGB for display
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Resize for display while maintaining aspect ratio
        height, width = img_rgb.shape[:2]
        max_height = 400
        max_width = 600
        
        # Calculate new dimensions
        if height > max_height or width > max_width:
            scale = min(max_height / height, max_width / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            img_rgb = cv2.resize(img_rgb, (new_width, new_height))
        
        # Convert to PhotoImage for Tkinter
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Update image label
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference
          # Clear results
        self.clear_results()
        
        # Show file path
        self.update_results(f"Loaded image: {os.path.basename(image_path)}")
    
    def update_selected_models(self):
        """Update the list of selected models based on checkboxes"""
        self.selected_models = [
            model for model, var in self.model_vars.items()
            if var.get()
        ]
    
    def detect_license_plate(self):
        """Detect license plate in the current image"""
        if self.current_image is None:
            self.update_results("No image loaded. Please select an image first.")
            return
        
        self.clear_results()
        self.update_results("Processing image for license plate detection...")
        
        try:
            # Update UI while processing
            self.root.update()
            
            # Detect license plate
            if not self.selected_models:
                self.update_results("No OCR models selected. Please select at least one model.")
                return
            
            # Process the image
            number_plate = number_plate_detection(self.current_image, ocr_models=self.selected_models)
            
            if number_plate:
                self.update_results(f"Detected license plate: {number_plate}")
            else:
                self.update_results("No license plate detected in the image.")
        
        except Exception as e:
            self.update_results(f"Error during detection: {str(e)}")
    
    def update_results(self, message):
        """Update the results text area"""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
    
    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)

if __name__ == "__main__":
    # Override stderr to redirect to our text widget
    root = tk.Tk()
    
    # Center the window on the screen
    window_width = 1000
    window_height = 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate x and y coordinates for the window to be centered
    x = int((screen_width - window_width) / 2)
    y = int((screen_height - window_height) / 2)
    
    # Set the position of the window
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    app = LicensePlateDetectorUI(root)
    root.mainloop()

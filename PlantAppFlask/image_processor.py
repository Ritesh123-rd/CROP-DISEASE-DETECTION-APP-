"""
PlantCare AI - OpenCV Image Processing Module
Features: Preprocessing, Enhancement, Edge Detection, Camera Capture
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64

class ImageProcessor:
    """OpenCV-based image processor for plant disease detection"""
    
    def __init__(self):
        self.target_size = (224, 224)
    
    def preprocess_image(self, image_bytes):
        """
        Preprocess image for better model prediction
        - Resize
        - Denoise
        - Color normalization
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Failed to load image"
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Denoise
        img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
        
        # Resize
        img_resized = cv2.resize(img_denoised, self.target_size)
        
        # Normalize
        img_normalized = img_resized / 255.0
        
        return img_normalized, None
    
    def enhance_image(self, image_bytes):
        """
        Enhance image quality
        - Contrast adjustment (CLAHE)
        - Sharpening
        - Color correction
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Failed to load image"
        
        # Convert to LAB color space for CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        img_sharpened = cv2.filter2D(img_enhanced, -1, kernel)
        
        return img_sharpened, None
    
    def detect_edges(self, image_bytes):
        """
        Detect edges in image using Canny edge detection
        Useful for leaf shape analysis
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Failed to load image"
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges, None
    
    def segment_leaf(self, image_bytes):
        """
        Segment leaf from background using color-based segmentation
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, None, "Failed to load image"
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define green color range for leaf detection
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to image
        result = cv2.bitwise_and(img, img, mask=mask)
        
        return result, mask, None
    
    def analyze_leaf_health(self, image_bytes):
        """
        Analyze leaf color distribution to detect potential diseases
        Returns health metrics based on color analysis
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Failed to load image"
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        # Analyze color distribution
        # Green hue range: 35-85
        green_ratio = np.sum(h_hist[35:85]) / np.sum(h_hist)
        
        # Yellow/Brown hue range: 15-35 (potential disease indicator)
        yellow_ratio = np.sum(h_hist[15:35]) / np.sum(h_hist)
        
        # Calculate health score (0-100)
        health_score = float(min(100, max(0, green_ratio * 100 - yellow_ratio * 50)))
        
        analysis = {
            "health_score": round(health_score, 2),
            "green_percentage": round(float(green_ratio * 100), 2),
            "yellow_brown_percentage": round(float(yellow_ratio * 100), 2),
            "potential_issue": bool(yellow_ratio > 0.3)
        }
        
        return analysis, None
    
    def get_confidence_threshold(self, predictions):
        """
        Check if prediction confidence is high enough
        Returns True if known plant, False if likely unknown
        """
        max_confidence = np.max(predictions)
        
        # If max confidence is below 60%, likely unknown plant
        if max_confidence < 0.6:
            return False, max_confidence
        return True, max_confidence
    
    def numpy_to_base64(self, img_array):
        """Convert numpy array to base64 for web display"""
        if len(img_array.shape) == 2:
            # Grayscale
            img_pil = Image.fromarray(img_array)
        else:
            # Color (BGR to RGB)
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"


# Global instance
image_processor = ImageProcessor()

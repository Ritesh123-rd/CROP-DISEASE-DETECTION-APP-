"""
PlantCare AI - Prediction Service with OpenCV
Loads trained model and predicts plant diseases
Uses OpenCV for image preprocessing and enhancement
"""

import os
import json
import numpy as np
from PIL import Image
import io

# Lazy imports
model = None
class_names = None

def load_model():
    """Load the trained model and class names"""
    global model, class_names
    
    if model is None:
        import tensorflow as tf
        MODEL_PATH = 'model/plant_disease_model.h5'
        CLASSES_PATH = 'model/classes.json'
        
        if os.path.exists(MODEL_PATH):
            print("🔄 Loading trained model...")
            model = tf.keras.models.load_model(MODEL_PATH)
            
            with open(CLASSES_PATH, 'r') as f:
                class_names = json.load(f)
            
            print("✅ Model loaded successfully!")
            return True
        else:
            print("⚠️ Model not found. Please train the model first.")
            return False
    
    return True

def preprocess_with_opencv(image_bytes):
    """Preprocess image using OpenCV for better predictions"""
    try:
        from image_processor import image_processor
        
        # Enhance image first
        enhanced, err = image_processor.enhance_image(image_bytes)
        if err:
            return None, err
        
        # Preprocess for model
        processed, err = image_processor.preprocess_image(image_bytes)
        if err:
            return None, err
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        return processed, None
        
    except ImportError:
        # Fallback to PIL if OpenCV not available
        return preprocess_image_pil(image_bytes), None

def preprocess_image_pil(image_bytes):
    """Fallback preprocessing with PIL"""
    IMG_SIZE = 224
    
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def analyze_with_opencv(image_bytes):
    """Get additional analysis using OpenCV"""
    try:
        from image_processor import image_processor
        
        # Get health analysis
        analysis, err = image_processor.analyze_leaf_health(image_bytes)
        if err:
            return None
        
        return analysis
    except:
        return None

def predict_disease(image_bytes):
    """Predict disease from image bytes with OpenCV preprocessing"""
    
    if not load_model():
        return {
            "success": False,
            "error": "Model not loaded. Please train the model first."
        }
    
    try:
        # OpenCV preprocessing
        img_array, err = preprocess_with_opencv(image_bytes)
        if err:
            # Fallback to simple preprocessing
            img_array = preprocess_image_pil(image_bytes)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get top prediction
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Check if confidence is too low (likely unknown plant)
        if confidence < 0.5:
            # Get OpenCV health analysis
            cv_analysis = analyze_with_opencv(image_bytes)
            
            return {
                "success": True,
                "plantName": "Unknown Plant",
                "disease": "Not in database",
                "confidence": f"{confidence * 100:.1f}%",
                "treatment": "This plant is not in our database. The model was trained on Tomato, Potato, and Pepper plants only. Please upload an image of these plants for accurate detection.",
                "isUnknown": True,
                "healthAnalysis": cv_analysis
            }
        
        # Get class name
        disease_name = class_names[str(predicted_class_idx)]
        
        # Parse plant and disease
        parts = disease_name.replace('___', '_').replace('__', '_').split('_')
        plant_name = parts[0] if parts else "Unknown"
        
        # Get OpenCV health analysis
        cv_analysis = analyze_with_opencv(image_bytes)
        
        if 'healthy' in disease_name.lower():
            disease = "Healthy"
            treatment = "Your plant looks healthy! Keep maintaining good care practices."
        else:
            disease = ' '.join(parts[1:]) if len(parts) > 1 else disease_name
            treatment = get_treatment(disease_name)
        
        return {
            "success": True,
            "plantName": f"{plant_name} Leaf",
            "disease": disease,
            "confidence": f"{confidence * 100:.1f}%",
            "treatment": treatment,
            "rawClass": disease_name,
            "isUnknown": False,
            "healthAnalysis": cv_analysis
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def get_treatment(disease_name):
    """Get treatment recommendation based on disease"""
    
    treatments = {
        'Tomato_Early_blight': 'Remove affected leaves, apply copper-based fungicide, ensure proper spacing between plants for air circulation.',
        'Tomato_Late_blight': 'Remove and destroy infected plants, apply fungicide containing chlorothalonil, improve drainage.',
        'Tomato_Bacterial_spot': 'Remove infected leaves, apply copper-based bactericide, avoid overhead watering.',
        'Tomato_Leaf_Mold': 'Improve ventilation, reduce humidity, apply fungicide, remove affected leaves.',
        'Tomato_Septoria_leaf_spot': 'Remove infected leaves, apply fungicide, avoid overhead watering, rotate crops.',
        'Tomato_Spider_mites': 'Spray with insecticidal soap or neem oil, increase humidity, introduce predatory mites.',
        'Tomato__Target_Spot': 'Apply fungicide, remove infected plant debris, ensure good air circulation.',
        'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Control whitefly population, remove infected plants, use resistant varieties.',
        'Tomato__Tomato_mosaic_virus': 'Remove infected plants, sanitize tools, use resistant varieties.',
        'Potato___Early_blight': 'Apply fungicide, remove infected leaves, practice crop rotation.',
        'Potato___Late_blight': 'Apply fungicide immediately, destroy infected plants, improve drainage.',
        'Pepper__bell___Bacterial_spot': 'Apply copper-based spray, remove infected leaves, avoid overhead watering.',
    }
    
    for key in treatments:
        if key.lower() in disease_name.lower():
            return treatments[key]
    
    return 'Consult a local agricultural expert for specific treatment recommendations.'

def get_edge_detection(image_bytes):
    """Get edge detection image for leaf analysis"""
    try:
        from image_processor import image_processor
        
        edges, err = image_processor.detect_edges(image_bytes)
        if err:
            return None
        
        # Convert to base64 for display
        return image_processor.numpy_to_base64(edges)
    except:
        return None

def get_enhanced_image(image_bytes):
    """Get enhanced image"""
    try:
        from image_processor import image_processor
        
        enhanced, err = image_processor.enhance_image(image_bytes)
        if err:
            return None
        
        return image_processor.numpy_to_base64(enhanced)
    except:
        return None

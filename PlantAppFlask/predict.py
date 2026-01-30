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
    """Get treatment recommendation based on disease with structured format"""
    
    treatments = {
        'Tomato_Early_blight': {
            'precautions': [
                'Remove and destroy affected leaves immediately',
                'Ensure proper spacing between plants for air circulation',
                'Avoid overhead watering - water at the base',
                'Practice crop rotation every 2-3 years',
                'Mulch around plants to prevent soil splash'
            ],
            'organic_manures': [
                'Neem Cake (नीम खली): Apply 100-150g per plant - natural fungicide properties',
                'Vermicompost (केंचुआ खाद): Apply 500g per plant - improves plant immunity',
                'Homemade Remedy: Mix 1 liter buttermilk (छाछ) with 10 liters water, spray weekly',
                'Trichoderma: Mix 50g in 1kg vermicompost, apply around roots'
            ],
            'inorganic_manures': [
                'Mancozeb 75% WP: 2-2.5g per liter water, spray every 10-15 days',
                'Copper Oxychloride 50% WP: 3g per liter water',
                'NPK 19:19:19: 5g per liter as foliar spray for plant strength'
            ]
        },
        'Tomato_Late_blight': {
            'precautions': [
                'Remove and destroy infected plants immediately',
                'Improve field drainage to reduce humidity',
                'Avoid planting in low-lying waterlogged areas',
                'Use disease-free seeds and seedlings',
                'Maintain proper plant spacing for ventilation'
            ],
            'organic_manures': [
                'Neem Oil: 5ml per liter water, spray every 7 days',
                'Cow Dung Slurry (गोबर घोल): 1kg in 10L water, ferment 3 days, dilute 1:5 and spray',
                'Wood Ash (राख): Sprinkle around base - reduces soil moisture and fungal growth',
                'Homemade Bordeaux Mix: 100g copper sulphate + 100g lime in 10L water'
            ],
            'inorganic_manures': [
                'Metalaxyl + Mancozeb (Ridomil Gold): 2.5g per liter water',
                'Chlorothalonil 75% WP: 2g per liter water, spray every 7-10 days',
                'Potassium Phosphonate: 3ml per liter as preventive spray'
            ]
        },
        'Tomato_Bacterial_spot': {
            'precautions': [
                'Remove infected leaves and fruits immediately',
                'Avoid overhead irrigation - use drip irrigation',
                'Do not work in field when plants are wet',
                'Sanitize all tools with 10% bleach solution',
                'Use certified disease-free seeds'
            ],
            'organic_manures': [
                'Pseudomonas fluorescens: 10g per liter, spray every 10 days',
                'Neem Cake: 150g per plant at base',
                'Homemade Garlic Spray: Blend 100g garlic in 1L water, strain, dilute 1:10 and spray',
                'Turmeric Powder: 5g mixed in 1L water, spray weekly'
            ],
            'inorganic_manures': [
                'Copper Hydroxide 77% WP: 2g per liter water',
                'Streptomycin Sulphate: 0.5g per liter water (use carefully)',
                'Copper Oxychloride: 3g per liter, spray every 7 days'
            ]
        },
        'Tomato_Leaf_Mold': {
            'precautions': [
                'Improve greenhouse/field ventilation',
                'Reduce humidity below 85%',
                'Remove lower leaves touching soil',
                'Avoid excess nitrogen fertilizer',
                'Space plants adequately'
            ],
            'organic_manures': [
                'Trichoderma viride: 5g per liter, spray on leaves',
                'Neem Oil: 3ml per liter with liquid soap',
                'Homemade Baking Soda Spray: 10g baking soda + 5ml oil in 1L water',
                'Vermicompost Tea: Soak 1kg vermicompost in 10L water for 24hrs, strain and spray'
            ],
            'inorganic_manures': [
                'Carbendazim 50% WP: 1g per liter water',
                'Mancozeb: 2.5g per liter, alternate with other fungicides',
                'Sulphur 80% WP: 2g per liter (avoid in hot weather)'
            ]
        },
        'Tomato_Septoria_leaf_spot': {
            'precautions': [
                'Remove infected lower leaves immediately',
                'Avoid overhead watering',
                'Practice 2-3 year crop rotation',
                'Destroy crop residue after harvest',
                'Use mulch to prevent soil splash'
            ],
            'organic_manures': [
                'Neem Cake: 100g per plant, mix in soil',
                'Trichoderma: Apply with organic manure 50g/kg',
                'Homemade: Diluted cow urine (गोमूत्र) 1:10, spray weekly',
                'Panchagavya: 30ml per liter water, spray every 15 days'
            ],
            'inorganic_manures': [
                'Mancozeb 75% WP: 2.5g per liter water',
                'Chlorothalonil: 2g per liter, spray every 7-10 days',
                'Azoxystrobin: 1ml per liter water'
            ]
        },
        'Tomato_Spider_mites': {
            'precautions': [
                'Increase humidity around plants',
                'Remove heavily infested leaves',
                'Avoid dusty conditions near plants',
                'Introduce natural predators like ladybugs',
                'Check undersides of leaves regularly'
            ],
            'organic_manures': [
                'Neem Oil: 5ml per liter + liquid soap, spray undersides of leaves',
                'Homemade Chilli-Garlic Spray: 50g chilli + 50g garlic, blend in 1L water, strain, spray',
                'Soap Solution: 5g liquid soap per liter water',
                'Onion Extract: Blend 100g onion in 1L water, ferment 24hrs, strain and spray'
            ],
            'inorganic_manures': [
                'Dicofol 18.5% EC: 2.5ml per liter water',
                'Abamectin 1.8% EC: 0.5ml per liter water',
                'Sulphur 80% WP: 3g per liter (as miticide)'
            ]
        },
        'Tomato__Target_Spot': {
            'precautions': [
                'Remove infected plant debris',
                'Ensure good air circulation',
                'Avoid continuous tomato cultivation',
                'Water at base of plants only',
                'Apply preventive sprays during humid weather'
            ],
            'organic_manures': [
                'Trichoderma harzianum: 5g per liter spray',
                'Neem Oil: 3-5ml per liter water',
                'Homemade: Papaya leaf extract - blend 100g leaves in 1L water, spray',
                'Cow Dung + Neem: Fermented mix for root application'
            ],
            'inorganic_manures': [
                'Azoxystrobin 23% SC: 1ml per liter water',
                'Mancozeb: 2.5g per liter as preventive',
                'Propiconazole 25% EC: 1ml per liter water'
            ]
        },
        'Tomato__Tomato_YellowLeaf__Curl_Virus': {
            'precautions': [
                'Control whitefly population (main carrier)',
                'Remove and destroy infected plants',
                'Use insect-proof nets in nursery',
                'Plant resistant varieties if available',
                'Avoid planting near older infected crops'
            ],
            'organic_manures': [
                'Yellow Sticky Traps: Install 10 per acre to trap whiteflies',
                'Neem Oil: 5ml per liter, spray every 5-7 days to control whitefly',
                'Homemade: Marigold border plantation - repels whiteflies naturally',
                'Beauveria bassiana: Bio-agent spray 5g per liter'
            ],
            'inorganic_manures': [
                'Imidacloprid 17.8% SL: 0.3ml per liter (for whitefly control)',
                'Thiamethoxam 25% WG: 0.3g per liter',
                'No cure for virus - focus on vector control and plant nutrition NPK 19:19:19'
            ]
        },
        'Tomato__Tomato_mosaic_virus': {
            'precautions': [
                'Remove and destroy infected plants',
                'Sanitize all tools with milk solution or bleach',
                'Wash hands before handling plants',
                'Use virus-free seeds from reliable source',
                'Control aphids which spread the virus'
            ],
            'organic_manures': [
                'Milk Spray: 1 part milk to 9 parts water - inactivates virus on contact',
                'Neem Oil: 5ml per liter to control aphids',
                'Vermicompost: Heavy application 1kg per plant for plant immunity',
                'Homemade: Tulsi (basil) leaf extract spray - antiviral properties'
            ],
            'inorganic_manures': [
                'No chemical cure for virus',
                'Imidacloprid: 0.3ml per liter for aphid control',
                'Foliar NPK: 5g per liter for plant strength',
                'Micronutrient spray: Zn + Mn + Fe for immunity'
            ]
        },
        'Potato___Early_blight': {
            'precautions': [
                'Remove infected leaves promptly',
                'Practice 3-year crop rotation',
                'Avoid excess nitrogen fertilizer',
                'Hill up potatoes properly',
                'Ensure good field drainage'
            ],
            'organic_manures': [
                'Neem Cake: 200g per plant mixed in soil',
                'Trichoderma viride: 50g per kg FYM, apply at planting',
                'Homemade: Whey (पनीर का पानी) diluted 1:5, spray weekly',
                'Vermicompost: 500g per plant for disease resistance'
            ],
            'inorganic_manures': [
                'Mancozeb 75% WP: 2.5g per liter, spray every 10-12 days',
                'Chlorothalonil: 2g per liter water',
                'Potash (MOP): 50g per plant to strengthen tubers'
            ]
        },
        'Potato___Late_blight': {
            'precautions': [
                'Act immediately - this disease spreads fast!',
                'Destroy infected plants and tubers',
                'Improve drainage urgently',
                'Avoid irrigation during humid weather',
                'Use resistant varieties like Kufri Jyoti'
            ],
            'organic_manures': [
                'Bordeaux Mixture: 1% solution (1kg CuSO4 + 1kg lime in 100L water)',
                'Trichoderma: Emergency application 10g per liter',
                'Homemade: Garlic + Neem double spray every 3 days',
                'Wood Ash: Heavy application around plants'
            ],
            'inorganic_manures': [
                'Metalaxyl + Mancozeb (Ridomil Gold): 2.5g per liter - most effective',
                'Cymoxanil + Mancozeb: 3g per liter water',
                'Dimethomorph: 1g per liter in severe cases'
            ]
        },
        'Pepper__bell___Bacterial_spot': {
            'precautions': [
                'Remove infected plant parts immediately',
                'Avoid working with wet plants',
                'Use drip irrigation only',
                'Sanitize tools between plants',
                'Use disease-free seedlings'
            ],
            'organic_manures': [
                'Pseudomonas fluorescens: 10g per liter, spray every 7 days',
                'Neem Oil + Turmeric: 5ml neem + 5g turmeric per liter',
                'Homemade Aloe Vera Spray: Blend 100g aloe gel in 1L water, spray',
                'Trichoderma soil drench: 5g per liter at root zone'
            ],
            'inorganic_manures': [
                'Copper Hydroxide 77% WP: 2g per liter water',
                'Streptocycline: 0.5g + Copper oxychloride 3g per liter',
                'Kasugamycin: 2ml per liter for severe infection'
            ]
        }
    }
    
    # Format the treatment into readable text
    for key in treatments:
        if key.lower() in disease_name.lower():
            t = treatments[key]
            result = "📋 PRECAUTIONS TO BE FOLLOWED:\n"
            for i, p in enumerate(t['precautions'], 1):
                result += f"   {i}. {p}\n"
            
            result += "\n🌿 ORGANIC MANURES (जैविक खाद):\n"
            for i, o in enumerate(t['organic_manures'], 1):
                result += f"   {i}. {o}\n"
            
            result += "\n🧪 INORGANIC MANURES (रासायनिक खाद):\n"
            for i, io in enumerate(t['inorganic_manures'], 1):
                result += f"   {i}. {io}\n"
            
            return result
    
    return '''📋 PRECAUTIONS TO BE FOLLOWED:
   1. Consult a local agricultural expert
   2. Take clear photos of affected parts
   3. Isolate infected plants from healthy ones

🌿 ORGANIC MANURES (जैविक खाद):
   1. Neem Cake: 100-150g per plant
   2. Vermicompost: 500g per plant
   3. Cow Dung Manure: Well decomposed, 2kg per plant

🧪 INORGANIC MANURES (रासायनिक खाद):
   1. Consult local agricultural officer for specific recommendations
   2. NPK 19:19:19: 5g per liter as general foliar spray'''

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

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_cors import CORS
from functools import wraps
import os

app = Flask(__name__)
app.secret_key = 'plantcare-ai-secret-key-2026'
CORS(app)

# Initialize database
from database import init_db, create_user, login_user, save_diagnosis, get_diagnosis_history, save_chat, get_diagnosis_stats, get_user_by_email, reset_password
init_db()

# Upload folder for images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== Authentication Decorator =====
def login_required(f):
    """Decorator to protect routes - requires login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # For API routes, return JSON error
            if request.path.startswith('/api/'):
                return jsonify({"success": False, "error": "Please login first"}), 401
            # For page routes, redirect to login
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ===== Public Routes (no login required) =====

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login')
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot_password.html')

# ===== Protected Routes (login required) =====

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/diagnosis')
@login_required
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@app.route('/history')
@login_required
def history():
    return render_template('history.html')

# ===== API Endpoints =====

@app.route('/api/signup', methods=['POST'])
def api_signup():
    """User signup with database storage"""
    data = request.json
    name = data.get('name', '')
    email = data.get('email', '')
    password = data.get('password', '')
    phone = data.get('phone', '')
    dob = data.get('dob', '')
    gender = data.get('gender', '')
    
    if not email or not password:
        return jsonify({"success": False, "error": "Email and password required"})
    
    result = create_user(name, email, password, phone, dob, gender)
    return jsonify(result)

@app.route('/api/login', methods=['POST'])
def api_login():
    """User login with database verification"""
    data = request.json
    email = data.get('email', '')
    password = data.get('password', '')
    
    result = login_user(email, password)
    
    if result['success']:
        session['user_id'] = result['user']['id']
        session['user_name'] = result['user']['name']
        session['user_email'] = result['user']['email']
    
    return jsonify(result)

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """User logout"""
    session.clear()
    return jsonify({"success": True, "message": "Logged out"})

@app.route('/api/reset-password', methods=['POST'])
def api_reset_password():
    """Reset user password"""
    data = request.json
    email = data.get('email', '')
    new_password = data.get('newPassword', '')
    
    if not email or not new_password:
        return jsonify({"success": False, "error": "Email and new password required"})
    
    if len(new_password) < 4:
        return jsonify({"success": False, "error": "Password must be at least 4 characters"})
    
    result = reset_password(email, new_password)
    return jsonify(result)

@app.route('/api/user', methods=['GET'])
@login_required
def api_user():
    """Get current user info"""
    return jsonify({
        "success": True,
        "user": {
            "id": session.get('user_id'),
            "name": session.get('user_name'),
            "email": session.get('user_email')
        }
    })

@app.route('/api/analyze', methods=['POST'])
@login_required
def api_analyze():
    """Analyze plant image using Gemini Vision API with database storage"""
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image provided"})
            
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Save image locally
        image_filename = f"scan_{len(os.listdir(UPLOAD_FOLDER))}_{session.get('user_id')}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        # Gemini Vision Integration
        import google.generativeai as genai
        from dotenv import load_dotenv
        import PIL.Image
        import io
        import json
        
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
             return jsonify({"success": False, "error": "Gemini API Key not found"})

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Prepare image for Gemini
        img = PIL.Image.open(io.BytesIO(image_bytes))
        
        # Optimize image size (Resize if larger than 1024x1024 to prevent OOM/Timeouts)
        img.thumbnail((1024, 1024))

        # Prompt for analysis
        prompt = """
        Analyze this image of a plant leaf. return a JSON object with the following fields:
        1. plantName: The name of the plant.
        2. disease: The name of the disease or "Healthy" if no disease is found.
        3. confidence: A percentage string (e.g., "95%") indicating confidence in the diagnosis.
        4. treatment: A brief, helpful treatment recommendation if diseased, or a care tip if healthy.
        5. health_score: A number from 0 to 100 representing the plant's health (100 is perfectly healthy).
        6. isUnknown: Boolean, set to true ONLY if the image is clearly NOT a plant.
        
        Output ONLY the raw JSON string.
        """
        
        gemini_response = model.generate_content([prompt, img])
        
        # Debug Logging
        print("Debugging Gemini Response:")
        try:
            print(f"Safety Ratings: {gemini_response.prompt_feedback}")
        except:
            pass
            
        try:
            text_response = gemini_response.text.strip()
            print(f"Raw Text Response: {text_response}")
        except Exception as e:
             print(f"Error accessing .text: {e}")
             # Check if blocked
             if gemini_response.candidates and gemini_response.candidates[0].finish_reason:
                 return jsonify({"success": False, "error": f"Image blocked due to safety filters ({gemini_response.candidates[0].finish_reason})"})
             return jsonify({"success": False, "error": "No text returned from AI"})

        # Clean up any markdown formatting if present
        if text_response.startswith('```json'):
            text_response = text_response[7:]
        if text_response.endswith('```'):
            text_response = text_response[:-3]
            
        try:
            result = json.loads(text_response)
            result['success'] = True
        except json.JSONDecodeError as je:
            print(f"JSON Decode Error: {je}")
            return jsonify({"success": False, "error": f"Failed to parse AI response: {str(je)}"})

        # Save to database
        user_id = session.get('user_id')
        save_diagnosis(
            user_id=user_id,
            plant_name=result.get('plantName', 'Unknown'),
            disease=result.get('disease', 'Unknown'),
            confidence=result.get('confidence', '0%'),
            treatment=result.get('treatment', ''),
            image_path=image_path,
            health_score=result.get('health_score'),
            is_unknown=result.get('isUnknown', False)
        )
        
        return jsonify(result)

    except Exception as e:
        print(f"Analysis Error Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    """Get diagnosis history from database"""
    user_id = session.get('user_id')
    history = get_diagnosis_history(user_id, limit=20)
    return jsonify({"success": True, "history": history})

@app.route('/api/stats', methods=['GET'])
@login_required
def api_stats():
    """Get diagnosis statistics"""
    user_id = session.get('user_id')
    stats = get_diagnosis_stats(user_id)
    return jsonify({"success": True, "stats": stats})

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    """Chat endpoint with database storage"""
    data = request.json
    message = data.get('message', '')
    
    # Gemini Chat Integration
    try:
        import google.generativeai as genai
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            # Plant Expert Persona - Relaxed
            prompt = f"""You are a helpful and knowledgeable Plant Expert AI. 
            User's question: {message}
            
            Provide a helpful, accurate, and friendly response. 
            You can give detailed advice but keep it easy to understand.
            If the question is not about plants, gardening, or agriculture, politely steer the conversation back to plants.
            """
            
            gemini_response = model.generate_content(prompt)
            response = gemini_response.text
        else:
            response = "Error: Gemini API key not configured."
            
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        # Fallback to simple logic if API fails
        response = "I'm having trouble connecting to my AI brain right now. Please try again in a moment."

    # Save to database
    user_id = session.get('user_id')
    save_chat(user_id, message, response)
    
    return jsonify({"success": True, "response": response})

if __name__ == '__main__':
    print("🌿 PlantCare AI Server Starting...")
    print("🔐 Authentication: Enabled")
    print("📱 Open http://localhost:5000 in your browser")
    print("📱 For mobile: http://<your-ip>:5000")
    print("💾 Database: SQLite (plantcare.db)")
    app.run(debug=True, host='0.0.0.0', port=5000)

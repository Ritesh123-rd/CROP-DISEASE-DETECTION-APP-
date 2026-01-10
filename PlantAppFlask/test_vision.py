import google.generativeai as genai
import os
from dotenv import load_dotenv
import PIL.Image
import json

# Setup
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-3-flash-preview')

# Load Image
img_path = r"dataset_mini/Pepper__bell___healthy/00100ffa-095e-4881-aebf-61fe5af7226e___JR_HL 7886.JPG"
if not os.path.exists(img_path):
    print(f"Error: Image not found at {img_path}")
    exit(1)

img = PIL.Image.open(img_path)
print(f"Image loaded: {img.size}")

# Prompt
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

print("Sending request to Gemini...")
try:
    response = model.generate_content([prompt, img])
    print("\n--- Raw Response ---")
    print(response.text)
    print("--- End Raw Response ---\n")
    
    # Parse
    text_response = response.text.strip()
    if text_response.startswith('```json'):
        text_response = text_response[7:]
    if text_response.endswith('```'):
        text_response = text_response[:-3]
    
    result = json.loads(text_response)
    print("\nParsed JSON:")
    print(json.dumps(result, indent=2))

except Exception as e:
    print(f"\nERROR: {str(e)}")

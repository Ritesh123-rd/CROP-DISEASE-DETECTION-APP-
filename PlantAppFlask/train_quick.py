"""
PlantCare AI - QUICK Training Script
Uses only 5 images per class for fast demo training
"""

import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
DATASET_PATH = 'dataset/PlantVillage'
MINI_DATASET_PATH = 'dataset_mini'
MODEL_PATH = 'model/plant_disease_model.h5'
CLASSES_PATH = 'model/classes.json'

IMAGES_PER_CLASS = 5  # 5 images per class for quick demo

def create_mini_dataset():
    """Create mini dataset with few images per class for quick training"""
    
    print("📂 Creating mini dataset for quick training...")
    
    # Remove old mini dataset if exists
    if os.path.exists(MINI_DATASET_PATH):
        shutil.rmtree(MINI_DATASET_PATH)
    
    os.makedirs(MINI_DATASET_PATH, exist_ok=True)
    
    # Get all class folders
    classes = [d for d in os.listdir(DATASET_PATH) 
               if os.path.isdir(os.path.join(DATASET_PATH, d)) and d != 'PlantVillage']
    
    print(f"📊 Found {len(classes)} disease classes")
    
    total_images = 0
    for class_name in classes:
        class_src = os.path.join(DATASET_PATH, class_name)
        class_dst = os.path.join(MINI_DATASET_PATH, class_name)
        os.makedirs(class_dst, exist_ok=True)
        
        # Get image files
        images = [f for f in os.listdir(class_src) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Copy only N images
        for img in images[:IMAGES_PER_CLASS]:
            src = os.path.join(class_src, img)
            dst = os.path.join(class_dst, img)
            shutil.copy2(src, dst)
            total_images += 1
        
        print(f"  ✅ {class_name}: {min(len(images), IMAGES_PER_CLASS)} images")
    
    print(f"\n📦 Mini dataset created: {total_images} images total")
    return total_images

def create_model(num_classes):
    """Create simple CNN model for quick training"""
    
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_quick():
    """Quick training with minimal data"""
    
    print("🌿 PlantCare AI - QUICK Model Training")
    print("=" * 50)
    
    # Create mini dataset
    create_mini_dataset()
    
    # Data generator - no validation split for small dataset
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Training data
    train_gen = datagen.flow_from_directory(
        MINI_DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    num_classes = len(train_gen.class_indices)
    print(f"\n📊 Classes: {num_classes}")
    print(f"📈 Training: {train_gen.samples} images")
    
    # Save class names
    os.makedirs('model', exist_ok=True)
    class_names = {v: k for k, v in train_gen.class_indices.items()}
    with open(CLASSES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"💾 Classes saved to {CLASSES_PATH}")
    
    # Create and train model
    print("\n🔧 Creating simple CNN model...")
    model = create_model(num_classes)
    
    print("\n🚀 Training (quick mode)...")
    model.fit(train_gen, epochs=EPOCHS)
    
    # Save model
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"📦 Model size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
    
    print("\n🎉 Quick training complete!")
    print("   Run 'python app.py' to use the model")

if __name__ == '__main__':
    train_quick()

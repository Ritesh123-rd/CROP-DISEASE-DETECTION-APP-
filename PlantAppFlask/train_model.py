"""
PlantCare AI - ML Model Training Script
Uses TensorFlow/Keras CNN for Plant Disease Detection
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = 'dataset/PlantVillage'
MODEL_PATH = 'model/plant_disease_model.h5'
CLASSES_PATH = 'model/classes.json'

# Disease classes to use (4 main categories for demo)
SELECTED_CLASSES = [
    'Tomato_Early_blight',
    'Tomato_Late_blight', 
    'Tomato_healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy'
]

def create_model(num_classes):
    """Create CNN model using MobileNetV2 transfer learning"""
    
    # Use MobileNetV2 as base (pre-trained on ImageNet)
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data():
    """Prepare training and validation data"""
    
    print("📂 Preparing dataset...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    # Training data
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation data
    val_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def train_model():
    """Main training function"""
    
    print("🌿 PlantCare AI - Model Training")
    print("=" * 50)
    
    # Prepare data
    train_gen, val_gen = prepare_data()
    
    num_classes = len(train_gen.class_indices)
    print(f"📊 Found {num_classes} disease classes")
    print(f"📈 Training samples: {train_gen.samples}")
    print(f"📉 Validation samples: {val_gen.samples}")
    
    # Save class names
    os.makedirs('model', exist_ok=True)
    class_names = {v: k for k, v in train_gen.class_indices.items()}
    with open(CLASSES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"💾 Class names saved to {CLASSES_PATH}")
    
    # Create model
    print("\n🔧 Creating CNN model with MobileNetV2...")
    model = create_model(num_classes)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2
        )
    ]
    
    # Train
    print("\n🚀 Starting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # Save model
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")
    
    # Final accuracy
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\n📊 Final Validation Accuracy: {val_acc*100:.2f}%")
    
    return history

if __name__ == '__main__':
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🎮 GPU available: {gpus}")
    else:
        print("💻 Running on CPU")
    
    train_model()

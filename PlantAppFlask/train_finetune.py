"""
PlantCare AI - Model Fine-Tuning Script
Loads the pre-trained model and fine-tunes it for better accuracy
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = 'dataset/PlantVillage'
MODEL_PATH = 'model/plant_disease_model.h5'

def prepare_data():
    """Prepare training and validation data"""
    print("📂 Preparing dataset for fine-tuning...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def fine_tune():
    print("🌿 PlantCare AI - Model Fine-Tuning")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Please run train_model.py first.")
        return

    # Load existing model
    print("🔄 Loading existing model...")
    model = keras.models.load_model(MODEL_PATH)
    
    # Get the base model (MobileNetV2)
    # It's usually the first layer in our Sequential model
    base_model = model.layers[0]
    
    print("🔓 Unfreezing top layers of base model...")
    base_model.trainable = True
    
    # Freeze all layers except the last 30
    # MobileNetV2 has 155 layers total
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    print(f"   Freezing first {fine_tune_at} layers, training rest...")
    
    # Recompile with very low learning rate
    print("🔧 Recompiling with low learning rate (1e-5)...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Prepare data
    train_gen, val_gen = prepare_data()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='model/plant_disease_model_finetuned.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train
    print("\n🚀 Starting fine-tuning...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # Save final model (overwrite original)
    print("\n💾 Saving improved model...")
    model.save(MODEL_PATH)
    
    # Final accuracy
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\n📊 Final Accuracy after Fine-Tuning: {val_acc*100:.2f}%")

if __name__ == '__main__':
    fine_tune()

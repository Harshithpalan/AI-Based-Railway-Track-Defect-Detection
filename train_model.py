"""
Training script for Railway Track Defect Detection Model
This is a placeholder script that demonstrates how to train a model.
Replace with your actual training data and architecture.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2  # Defect vs Normal

def create_model():
    """Create a CNN model for defect detection"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the model with data augmentation"""
    # Create model
    model = create_model()
    model.summary()
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data only rescaled
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Replace these paths with your actual data directories
    # Structure should be:
    # data/train/defect/  (images of defective tracks)
    # data/train/normal/  (images of normal tracks)
    # data/validation/defect/
    # data/validation/normal/
    
    train_dir = 'data/train'
    val_dir = 'data/validation'
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE
        )
        
        # Save the model
        if not os.path.exists('models'):
            os.makedirs('models')
        
        model.save('models/defect_detection_model.h5')
        print("Model saved successfully!")
        
        return history
    else:
        print("Training data not found. Please organize your data as follows:")
        print("data/train/defect/")
        print("data/train/normal/")
        print("data/validation/defect/")
        print("data/validation/normal/")
        return None

if __name__ == '__main__':
    print("Starting model training...")
    history = train_model()
    
    if history:
        print("\nTraining completed!")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

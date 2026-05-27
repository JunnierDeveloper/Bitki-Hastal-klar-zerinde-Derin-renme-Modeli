
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc

def build_model(num_classes):
    base_model = keras.applications.Xception(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.2),
        layers.Lambda(lambda x: keras.applications.xception.preprocess_input(x)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model, base_model

if __name__ == "__main__":
    print("Model eğitim scripti hazır. Notebook üzerindeki parametrelerle çalıştırılabilir.")

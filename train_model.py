
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import gc
import os

def build_model(num_classes):
    base_model = keras.applications.Xception(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    base_model.trainable = False
    
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

def plot_and_save_performance(history_transfer, history_fine):
    acc = history_transfer.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history_transfer.history['val_accuracy'] + history_fine.history['val_accuracy']
    
    plt.figure(figsize=(12, 6))
    plt.plot(acc, label='Eğitim Doğruluğu')
    plt.plot(val_acc, label='Doğrulama Doğruluğu')
    plt.axvline(x=len(history_transfer.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning Başlangıcı')
    plt.title('Xception Model Performansı')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_performance.png')
    print("✅ Grafik 'training_performance.png' olarak kaydedildi.")

if __name__ == '__main__':
    print('Model eğitim ve fine-tuning süreci başlatılmaya hazır.')

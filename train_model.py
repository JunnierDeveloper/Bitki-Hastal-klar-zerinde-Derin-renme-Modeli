
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

def plot_history(h_transfer, h_fine):
    acc = h_transfer.history['accuracy'] + h_fine.history['accuracy']
    val_acc = h_transfer.history['val_accuracy'] + h_fine.history['val_accuracy']
    
    plt.figure(figsize=(10, 6))
    plt.plot(acc, label='Eğitim Doğruluğu')
    plt.plot(val_acc, label='Doğrulama Doğruluğu')
    plt.axvline(x=len(h_transfer.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning Başlangıcı')
    plt.title('Eğitim Süreci: Doğruluk Performansı')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_performance.png')
    print("✅ Performans grafiği 'training_performance.png' olarak kaydedildi.")

if __name__ == "__main__":
    print("--- Model Eğitim Scripti Başlatıldı ---")
    # Bu script notebook ortamındaki değişkenlere (train_ds, val_ds vb.) bağımlıdır.
    # Notebook içinde çalıştırılması veya veri yükleme kodlarının buraya eklenmesi gerekir.

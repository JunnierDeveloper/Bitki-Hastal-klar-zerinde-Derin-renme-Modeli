
# Bitki Hastalıkları Teşhis Sistemi (Deep Learning)

Bu proje, Xception mimarisi kullanılarak bitki yapraklarındaki hastalıkları %99 başarı oranıyla tespit eden bir derin öğrenme modelidir.

## 🎯 Projenin Amacı
Bu çalışma, tarım alanında verimliliği artırmak ve bitki hastalıklarını erken aşamada teşhis ederek doğru müdahale yöntemlerinin seçilmesine yardımcı olmak amacıyla geliştirilmiştir. Modern derin öğrenme teknikleri (Transfer Learning ve Fine-tuning) kullanılarak yüksek doğruluk hedeflenmiştir.

## 📊 Veri Seti Hakkında
Projede iki ana veri kaynağı birleştirilmiştir:
1. **PlantVillage Veri Seti:** 15 farklı sınıfı (Sağlıklı ve hastalıklı Domates, Patates, Biber yaprakları) içeren geniş bir katalogdur.
2. **PlantDec Veri Seti:** Ekstra örneklerle modelin genelleme yeteneği artırılmıştır.

## 🚀 Teknik Detaylar
- **Model:** Xception (Pre-trained on ImageNet)
- **Yöntem:** 
    - 1. Aşama: Transfer Learning (Son katman eğitimi)
    - 2. Aşama: Fine-tuning (Tüm mimarinin düşük öğrenme oranıyla eğitilmesi)
- **Başarı Oranı:** %99.13 (Validation Accuracy)
- **Kütüphaneler:** TensorFlow, Keras, Matplotlib, Kaggle API

## 🛠️ Kurulum ve Kullanım
1. Gereksinimleri yükleyin: `pip install tensorflow matplotlib numpy`
2. Modeli yüklemek için: `model = tf.keras.models.load_model('plant_model_finetuned.keras')`
3. Veri seti Kaggle üzerinden `andresmgs/plantdec` ve `emmarex/plantdisease` adreslerinden temin edilebilir.

## 📂 Dosya Yapısı
- `kaggle.json`: Veri setini indirmek için yapılandırma.
- `train_model.py`: Model eğitim kodlarının Python versiyonu.
- `plant_model_finetuned.keras`: Eğitilmiş en iyi model (Google Drive üzerinden erişilebilir).

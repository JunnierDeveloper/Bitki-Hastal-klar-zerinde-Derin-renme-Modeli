# 🧠 Derin Öğrenme Model Mimarisi ve Teknik Analiz Raporu

Bu belgede, **PlantGuard AI** yapay zeka sisteminde yaprak hastalıklarını teşhis etmek amacıyla kullanılan derin öğrenme mimarisinin katman yapılandırması ve çift aşamalı eğitim (training & fine-tuning) stratejisi teknik detaylarıyla açıklanmaktadır.

---

## 🏗️ 1. Model Katman Yapısı (Xception Tabanlı)

Modelimiz, TensorFlow ve Keras kütüphaneleri kullanılarak **Sıralı (Sequential)** bir yapıda inşa edilmiştir:

```python
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
```

### 🔍 Katman Analizleri:

1. **Giriş Katmanı (`layers.Input`):**
   * Giriş boyutu **224 x 224 piksel** ve **3 renk kanalına (RGB)** sahip görüntüler için ayarlanmıştır.

2. **Veri Artırma Katmanları (`RandomFlip` & `RandomRotation`):**
   * **`RandomFlip('horizontal_and_vertical')`:** Resimleri dikey ve yatay olarak rastgele simetrisini alır.
   * **`RandomRotation(0.2)`:** Görüntüleri rastgele %20 (yaklaşık 72 derece) oranında döndürür.
   * **Faydası:** Modelin yaprakların yönünden ve konumundan bağımsız olarak teşhis koyabilmesini sağlar. Eğitim verisinin çeşitliliğini artırarak aşırı öğrenmeyi (overfitting) kalıcı olarak engeller.

3. **Ön İşleme Katmanı (`Lambda`):**
   * **Görev:** Xception modelinin beklediği standart ön işlemesi olan, pikselleri `[-1.0, 1.0]` aralığına ölçekleme işlevini (`preprocess_input`) yerine getirir.
   * *Not:* Yerel uygulamada bu katman baypas edilerek ön işleme doğrudan NumPy düzeyinde optimize edilmiştir.

4. **Önceden Eğitilmiş Taban Mimari (Xception Base Model):**
   * **Derinlik ve Başarım:** ImageNet veri setiyle eğitilmiş olan **Xception** mimarisidir.
   * **Çalışma Prensibi:** Geleneksel evrişim katmanları yerine **Derinlemesine Ayrılabilir Evrişim (Depthwise Separable Convolutions)** yapısını kullanır. Bu sayede hem çok daha az parametre ile çalışır hem de uzaysal ilişkileri çok daha yüksek başarımla yakalar.

5. **Boyutsal İndirgeme (`GlobalAveragePooling2D`):**
   * Xception tabanının çıkışında oluşan 3 boyutlu matrisi (`7x7x2048`), uzaysal ortalama alarak tek boyutlu bir vektöre (`2048,`) indirger.
   * **Faydası:** Geleneksel `Flatten` katmanına kıyasla parametre sayısını milyonlarca adet azaltır, aşırı öğrenmeyi engeller ve hesaplama hızını artırır.

6. **Düzenlileştirme (`Dropout`):**
   * Her eğitim adımında nöronların %30'unu rastgele devre dışı bırakır (`rate=0.3`).
   * **Faydası:** Nöronların birbirine bağımlı öğrenmesini önler, modeli kendi kendine genelleme yapmaya zorlar.

7. **Çıkış Katmanı (`Dense` - Softmax):**
   * 15 farklı kategori için olasılık dağılımı üreten `Dense` katmanıdır. **Softmax** aktivasyon fonksiyonu sayesinde çıkışların toplamı her zaman 1.0 (yani %100) olur ve en yüksek olasılık teşhis sonucu olarak kabul edilir.

---

## 📈 2. Çift Aşamalı Eğitim Stratejisi

Modelin eğitimi, derin öğrenmenin en başarılı tekniklerinden olan **Aşamalı İnce Ayar (Fine-Tuning)** yöntemiyle yapılmıştır:

### ⏱️ Aşama 1: Transfer Öğrenme (0 - 20 Epoch)
* **Durum:** Xception taban modelinin tüm katmanları kilitlenmiştir (`base_model.trainable = False`).
* **Öğrenme Oranı (Learning Rate):** `1e-3` (Adam optimizer)
* **Amaç:** Xception'ın zaten bildiği nesne tanıma özelliklerini bozmadan, en sondaki sınıflandırma katmanını bizim 15 yaprak kategorimizle hızlıca uyumlu hale getirmek.

### 🔬 Aşama 2: İnce Ayar - Fine-Tuning (20 - 35 Epoch)
* **Durum:** Xception taban modelinin katmanlarının kilitleri açılmıştır (`base_model.trainable = True`).
* **Öğrenme Oranı (Learning Rate):** `1e-5` (Son derece düşük öğrenme oranı)
* **Amaç:** Xception'ın derinlerdeki ağırlıklarını bizim bitki yapraklarımıza özel ince detayları (damar yapısı, leke şekli vb.) öğrenecek şekilde hassasça güncellemek. Çok düşük öğrenme oranı kullanılarak önceden öğrenilen genel bilginin bozulması (Catastrophic Forgetting) engellenmiştir.


# Bitki Hastalıkları Teşhis Sistemi (Derin Öğrenme)

## 📝 Proje Hakkında
Bu çalışma, tarımsal verimliliği artırmak amacıyla yaprak görüntülerinden bitki türünü ve üzerindeki hastalıkları tespit etmek için geliştirilmiştir. Proje kapsamında veri hazırlığı, model eğitimi ve GitHub entegrasyonu aşamaları tamamlanmıştır.

## 🚀 Teknik Başarılar
- **Model Mimarisi:** Xception (Transfer Learning + Fine-tuning)
- **Performans:** Yapılan optimizasyonlar ve düşük öğrenme oranı ile %99'un üzerinde doğruluk oranına ulaşılmıştır.
- **Görselleştirme:** Eğitim süreci sonunda doğruluk ve kayıp grafikleri otomatik olarak `training_performance.png` adıyla kaydedilir.

## 📁 Dosya Yapısı
- `train_model.py`: Modelin eğitim ve ince ayar süreçlerini yöneten ana script.
- `MODEL_DRIVE_LINK.txt`: Büyük boyutlu model dosyalarına erişim için Google Drive bağlantısı.
- `kaggle.json`: Veri setine erişim için gerekli yapılandırma.

## 🛠 Kurulum ve Kullanım
Eğitimi başlatmak için `python train_model.py` komutu kullanılabilir. Büyük `.keras` modelleri GitHub limitleri nedeniyle Drive üzerinde saklanmaktadır.

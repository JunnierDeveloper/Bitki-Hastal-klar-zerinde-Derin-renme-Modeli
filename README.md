# 🌿 Bitki Hastalıkları Teşhis ve Sınıflandırma Sistemi (PlantGuard AI)

Bu proje, **Xception** derin öğrenme mimarisi kullanılarak domates, patates ve biber yapraklarındaki 15 farklı sağlık ve hastalık durumunu teşhis eden yüksek başarımlı bir yapay zeka sistemidir. Proje kapsamında Google Colab üzerinde eğitilen model, modern ve premium tasarımlı bir **Streamlit** web uygulaması üzerinden yerel bilgisayarda kolayca çalıştırılabilmektedir.

---

## 🔗 Eğitilmiş Model Dosyaları (Google Drive)

GitHub'ın 100 MB dosya boyutu sınırı nedeniyle, eğittiğimiz `.keras` model dosyaları bu depoda doğrudan barındırılmamaktadır. Eğitilmiş model dosyalarımıza aşağıdaki tıklanabilir bağlantı üzerinden kolayca erişebilirsiniz:

👉 **[Google Drive Eğitilmiş Modelleri İndir](https://drive.google.com/drive/folders/1AoIRKp566gYCzvULiNnCL3D1YhZ99TQK?usp=sharing)**

### 📂 İndirilebilir Modeller:
1. **`plant_model_stage1.keras`** (Transfer Learning Aşaması - 20 Epoch)
2. **`plant_model_finetuned.keras`** (Fine-Tuning Aşaması - 35 Epoch)

> [!IMPORTANT]
> Uygulamanın çalışabilmesi için indirdiğiniz `.keras` dosyasını `derin öğrenme test.py` ile aynı klasöre kopyalamanız gerekmektedir.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gerekli Kütüphanelerin Kurulumu:
```bash
pip install tensorflow streamlit pillow numpy matplotlib
```

### 2. Uygulamayı Başlatma:
```bash
streamlit run "derin öğrenme test.py"
```

---

## 🎨 Uygulama Özellikleri
- **Akıllı Teşhis Portalı:** Yaprak fotoğraflarını sürükle-bırak yöntemiyle yükleyip anında sonuç alma.
- **Neon Tarama Efekti:** Görsel analiz edilirken yaprak üzerinde kayan neon analiz çizgisi animasyonu.
- **Organik Mücadele Kılavuzu:** Tespit edilen her hastalık için kimyasal kalıntı bırakmayan doğal ve organik tedavi reçeteleri.
- **Tehdit Seviyesi Göstergesi:** Hastalığın ciddiyetine göre renk kodlu risk uyarıları.
- **İnteraktif Olasılık Grafiği:** Modelin tüm sınıflar için tahmini olasılık dağılımını gösteren şık grafik.

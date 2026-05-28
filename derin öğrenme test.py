# -*- coding: utf-8 -*-
"""
PlantGuard AI - Yaprak Hastalığı Teşhis Uygulaması
Bu uygulama, Google Colab'da eğittiğiniz Xception tabanlı derin öğrenme modelini (.keras) 
kullanarak bitki yapraklarındaki hastalıkları anında teşhis eder ve organik tedavi önerileri sunar.
"""

import os
import sys

# TensorFlow, Keras ve oneDNN terminal log kalabalığını ve uyarılarını tamamen engelliyoruz
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # 1: INFO, 2: WARNING, 3: ERROR loglarını gizler
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'         # oneDNN optimizasyon uyarılarını gizler

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
try:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
except:
    pass

# 1. Eksik Kütüphane Kontrolü ve CLI Arayüzü
try:
    import streamlit as st
    import tensorflow as tf
    
    # Lambda katmanındaki 'keras' bağımlılığını global alanda tanımlıyoruz
    try:
        import keras
    except ImportError:
        from tensorflow import keras
        import sys
        sys.modules['keras'] = keras
        
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEP = str(e).split("'")[-2] if "'" in str(e) else str(e)

# Eğer kütüphaneler eksikse, kullanıcıya terminal üzerinden detaylı kurulum talimatları gösterilir.
if not HAS_DEPS:
    print("\n" + "="*70)
    print("🚨 HATA: Bazı gerekli kütüphaneler sisteminizde bulunamadı!")
    print(f"Eksik Kütüphane: {MISSING_DEP}")
    print("="*70)
    print("\nUygulamayı çalıştırmak için lütfen terminalde aşağıdaki komutu çalıştırarak")
    print("gerekli kütüphaneleri yükleyin:")
    print("\n👉  pip install tensorflow streamlit pillow numpy matplotlib")
    print("\nKütüphaneleri yükledikten sonra web arayüzünü başlatmak için:")
    print("\n👉  streamlit run \"dern öğrenme test.py\"")
    print("="*70 + "\n")
    
    # Minimal bir Python arayüzü ile hata mesajı gösterelim
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Kütüphane Eksik", 
            f"Gerekli kütüphaneler eksik (Örn: {MISSING_DEP}).\n\n"
            "Lütfen terminale gidip şu komutla yükleyin:\n"
            "pip install tensorflow streamlit pillow numpy matplotlib\n\n"
            "Ardından uygulamayı başlatmak için:\n"
            "streamlit run \"dern öğrenme test.py\""
        )
    except:
        pass
    sys.exit(1)

# --- SINIF VE VERİTABANI TANIMLARI ---

# Eğittiğiniz modelin çıktılarındaki 15 sınıfın tam ve alfabetik sırası
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Detaylı Hastalık, Belirti ve Organik Tedavi Veritabanı (Türkçe)
PLANT_DISEASES_DB = {
    'Pepper__bell___Bacterial_spot': {
        'name': 'Biber (Çan) - Bakteriyel Leke Hastalığı',
        'scientific_name': 'Xanthomonas campestris pv. vesicatoria',
        'severity': 'Yüksek Risk',
        'severity_color': '#FF9900',
        'symptoms': 'Yapraklarda küçük, su emmiş gibi görünen koyu lekeler oluşur. Bu lekelerin merkezleri zamanla kurur, kahverengiye döner ve etrafında sarı bir halka (hale) gelişir. Şiddetli enfeksiyonlarda yapraklar tamamen sararır ve dökülür.',
        'organic_treatments': [
            '**Bakır Uygulaması:** Hastalık başlangıcında ruhsatlı organik bakırlı fungisitler veya Bordo Bulamacı uygulayarak bakterilerin yayılmasını engelleyin.',
            '**Sanitasyon:** Enfekte olmuş yaprakları ve bitki kalıntılarını bahçeden hemen uzaklaştırın ve imha edin (kesinlikle kompost yapmayın).',
            '**Damlama Sulama:** Yaprakların kuru kalmasını sağlamak için üstten sulama yerine damlama sulama sistemleri tercih edin. Nem bakterilerin baş düşmanıdır.',
            '**Münavebe:** En az 2-3 yıl boyunca aynı alana biber, domates veya patates gibi Solanaceae ailesi üyelerini dikmeyin.'
        ]
    },
    'Pepper__bell___healthy': {
        'name': 'Biber (Çan) - Sağlıklı',
        'scientific_name': 'Capsicum annuum (Sağlıklı Bitki)',
        'severity': 'Sağlıklı',
        'severity_color': '#00cc66',
        'symptoms': 'Yapraklarda leke, sararma veya deformasyon bulunmamaktadır. Bitki dokusu canlı, yeşil rengi parlak ve gelişimi normal düzeydedir.',
        'organic_treatments': [
            '**Düzenli İzleme:** Bitkinin gelişimini haftalık olarak gözlemlemeye devam edin.',
            '**Bağışıklık Desteği:** Kompost veya sıvı solucan gübresi uygulayarak bitkinin doğal direncini yüksek tutun.',
            '**Doğru Sulama:** Toprak kurudukça, sabah erken saatlerde düzenli sulama yapın.',
            '**Faydalı Sprey:** Neem yağı (tespih ağacı yağı) ile periyodik hafif püskürtme yaparak koruyucu bir bariyer sağlayabilirsiniz.'
        ]
    },
    'Potato___Early_blight': {
        'name': 'Patates - Erken Yanıklık Hastalığı',
        'scientific_name': 'Alternaria solani',
        'severity': 'Orta Risk',
        'severity_color': '#FFCC00',
        'symptoms': 'Özellikle yaşlı alt yapraklarda eş merkezli halkalara sahip koyu kahverengi/siyah lekeler (hedef tahtası görünümü) oluşur. Lekelerin etrafında sarı alanlar görülür.',
        'organic_treatments': [
            '**Hava Sirkülasyonu:** Bitkilerin arasını geniş tutarak yaprakların hızla kurumasını sağlayın.',
            '**Alt Budama:** Toprağa temas eden veya yere çok yakın olan alt yaprakları budayarak sporların sıçramasını önleyin.',
            '**Organik Fungisit:** Bakır bazlı organik spreyler veya kekik/kekik yağı ekstraktı içeren doğal karışımlar kullanın.',
            '**Potasyum Takviyesi:** Topraktaki potasyum miktarını dengeli tutarak bitki dokusunu güçlendirin.'
        ]
    },
    'Potato___Late_blight': {
        'name': 'Patates - Geç Yanıklık Hastalığı (Mildiyö)',
        'scientific_name': 'Phytophthora infestans',
        'severity': 'Kritik Tehdit',
        'severity_color': '#FF3333',
        'symptoms': 'Yaprak uçlarında ve kenarlarında hızla büyüyen, su emmiş görünümlü, düzensiz koyu lekeler oluşur. Nemli havalarda yaprağın alt kısmında beyaz, kül benzeri küf örtüsü belirir.',
        'organic_treatments': [
            '**Sökme ve İmha:** Enfekte olmuş bitki kısımlarını veya ağır vakalarda tüm bitkiyi derhal sökün ve yakın/gömün.',
            '**Isırgan Otu Ekstraktı:** Doğal bir bağışıklık güçlendirici ve hafif fungisit olarak fermente ısırgan otu suyu püskürtün.',
            '**Nemi Azaltın:** Sulamayı kesinlikle damlama şeklinde yapın, yaprakları ıslatmaktan kaçının.',
            '**Erken Önlem:** Bölgede yağışlı ve nemli hava dalgası başlamadan önce koruyucu olarak organik bakırlı preparatlar uygulayın.'
        ]
    },
    'Potato___healthy': {
        'name': 'Patates - Sağlıklı',
        'scientific_name': 'Solanum tuberosum (Sağlıklı Bitki)',
        'severity': 'Sağlıklı',
        'severity_color': '#00cc66',
        'symptoms': 'Gövde dik, yapraklar tamamen yeşil, canlı ve lekesizdir. Gelişim normal seyrindedir.',
        'organic_treatments': [
            '**Malçlama:** Toprak yüzeyini saman veya kuru otlarla malçlayarak nem dengesini koruyun ve toprak sporlarının yapraklara sıçramasını engelleyin.',
            '**Organik Besleme:** Kül suyu veya kompost çayı ile bitkiyi sulayarak mineral desteği sağlayın.',
            '**Zararlı Kontrolü:** Patates böceğine karşı düzenli kontroller yapıp görülen böcekleri el ile toplayın.'
        ]
    },
    'Tomato_Bacterial_spot': {
        'name': 'Domates - Bakteriyel Leke Hastalığı',
        'scientific_name': 'Xanthomonas perforans',
        'severity': 'Yüksek Risk',
        'severity_color': '#FF9900',
        'symptoms': 'Yapraklarda, gövdede ve meyvede küçük, siyah-kahverengi dairesel lekeler. Yapraklar sararabilir, kıvrılabilir ve dökülebilir. Meyvelerde kabuksu çatlaklar oluşur.',
        'organic_treatments': [
            '**Bakır Püskürtme:** Organik sertifikalı bakır hidroksit veya Bordo bulamacı ile koruyucu ilaçlama yapın.',
            '**Nem Yönetimi:** Seralarda havalandırmayı artırın. Açık alanda sabah erken sulama yaparak gün boyu kuruluk sağlayın.',
            '**Alet Dezenfeksiyonu:** Budama makası gibi aletleri her bitkiden sonra %70 sirkeli veya alkollü su ile dezenfekte edin.',
            '**Temiz Tohum:** Bir sonraki sezon için kesinlikle hastalıklı bitkilerden tohum almayın.'
        ]
    },
    'Tomato_Early_blight': {
        'name': 'Domates - Erken Yanıklık Hastalığı',
        'scientific_name': 'Alternaria solani',
        'severity': 'Orta Risk',
        'severity_color': '#FFCC00',
        'symptoms': 'Yapraklarda koyu, halkalı (hedef tahtası benzeri) kahverengi lekeler. Genellikle ilk olarak alt yapraklarda başlar ve yukarı doğru yayılır.',
        'organic_treatments': [
            '**Yaprak Budaması:** Toprak seviyesinden yukarıya doğru ilk 30 cm\'lik alandaki tüm yaprakları budayarak toprak kökenli spor temasını kesin.',
            '**Kompost Çayı:** Bitki bağışıklığını ve yararlı mikrobiyal aktiviteyi artırmak için kompost çayı spreyleyin.',
            '**Kabartma Tozu Karışımı:** 1 litre suya 1 yemek kaşığı kabartma tozu (sodyum bikarbonat) ve birkaç damla organik sabun ekleyerek koruyucu sprey hazırlayın.'
        ]
    },
    'Tomato_Late_blight': {
        'name': 'Domates - Geç Yanıklık Hastalığı (Mildiyö)',
        'scientific_name': 'Phytophthora infestans',
        'severity': 'Kritik Tehdit',
        'severity_color': '#FF3333',
        'symptoms': 'Yapraklarda ve gövdede hızla yayılan büyük, soluk yeşilden kahverengiye dönen lekeler. Meyvelerde geniş, çökük kahverengi lekeler oluşur. Bitkiyi günler içinde kurutabilir.',
        'organic_treatments': [
            '**Enfekte Parçaları Sökün:** Hastalıklı yaprak, meyve ve gövdeleri kesip bahçeden çok uzak bir yere gömün.',
            '**Sarımsak Ekstraktı Spreyi:** Sarımsağın doğal antifungal özelliklerinden yararlanarak hazırladığınız sarımsaklı suyu bitkiye püskürtün.',
            '**Bakır Esaslı Çözümler:** Hastalık belirtisi görüldüğü an organik bakırlı fungisitleri prospektüse uygun uygulayın.',
            '**Güneşlenme:** Bitkileri budayarak güneş ışığının iç kısımlara girmesini sağlayın, bu nemi düşürür.'
        ]
    },
    'Tomato_Leaf_Mold': {
        'name': 'Domates - Yaprak Küfü Hastalığı',
        'scientific_name': 'Passalora fulva',
        'severity': 'Orta Risk',
        'severity_color': '#FFCC00',
        'symptoms': 'Yaprağın üst yüzeyinde soluk yeşil-sarı lekeler belirirken, tam altına denk gelen alt yüzeyde kadifemsi, zeytin yeşili veya morumsu bir küf tabakası oluşur.',
        'organic_treatments': [
            '**Bağıl Nem Kontrolü:** Seralarda nemi kesinlikle %85\'in altında tutun. Fanlar yardımıyla hava sirkülasyonu sağlayın.',
            '**Sık Dikimden Kaçının:** Bitkiler arasında rüzgar geçişine izin verecek mesafeler bırakın.',
            '**Kükürt Uygulaması:** Organik tarımda izin verilen toz veya ıslanabilir kükürt uygulamaları yaprak küfü gelişimini engeller.'
        ]
    },
    'Tomato_Septoria_leaf_spot': {
        'name': 'Domates - Septoria Yaprak Lekesi',
        'scientific_name': 'Septoria lycopersici',
        'severity': 'Orta Risk',
        'severity_color': '#FFCC00',
        'symptoms': 'Yapraklarda çok sayıda küçük, yuvarlak lekeler oluşur. Lekelerin merkezleri gri-beyaz, kenarları ise koyu kahverengidir. Merkezlerinde küçük siyah noktacıklar (piknid) görülür.',
        'organic_treatments': [
            '**Malç Kullanımı:** Toprak kökenli bu mantarın yağmur damlalarıyla yapraklara sıçramasını önlemek için kalın bir malç tabakası serin.',
            '**Budama:** Alt kısımdaki hasta yaprakları kesinlikle kuru havalarda budayın ve aletleri temizleyin.',
            '**Bitkiyi Kuru Tutun:** Yaprakları kesinlikle ıslatmayın, akşam sulamalarından kaçının.'
        ]
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'name': 'Domates - İki Noktalı Kırmızı Örümcek Zararı',
        'scientific_name': 'Tetranychus urticae',
        'severity': 'Orta-Yüksek Risk',
        'severity_color': '#FF9900',
        'symptoms': 'Yapraklarda çok küçük beyaz veya sarı noktacıklar (kloroz) oluşur. Yaprakların alt kısımlarında çok ince, ipeksi ağlar gözlenir. İlerleyen aşamalarda yapraklar kurur.',
        'organic_treatments': [
            '**Neem Yağı Spreyi:** Doğal bir böcek büyüme düzenleyici olan Neem yağı ve organik sıvı sabun karışımını yaprak altlarına tazyikli püskürtün.',
            '**Arap Sabunlu Su:** 1 litre suya 1 yemek kaşığı arap sabunu ekleyerek yaprakların altına uygulayın (Güneş altında uygulamayın, yaprakları yakabilir).',
            '**Faydalı Böcekler:** Kırmızı örümceklerin doğal düşmanı olan yırtıcı akarları (Phytoseiulus persimilis) bahçenize çekin veya salın.',
            '**Nemlendirme:** Kırmızı örümcekler kuru ve sıcak havayı sever. Bitki etrafını hafifçe nemlendirmek popülasyonu azaltır.'
        ]
    },
    'Tomato__Target_Spot': {
        'name': 'Domates - Hedef Leke Hastalığı',
        'scientific_name': 'Corynespora cassiicola',
        'severity': 'Orta Risk',
        'severity_color': '#FFCC00',
        'symptoms': 'Yapraklarda küçük dairesel lekeler oluşur. Lekeler büyüdükçe belirgin açık ve koyu kahverengi halkalar alır. Erken yanıklığa benzer ancak lekeler daha küçük ve daha dairesel sınırlıdır.',
        'organic_treatments': [
            '**Yabancı Ot Mücadelesi:** Tarla etrafındaki Solanaceae yabancı otlarını temizleyin çünkü virüs ve mantarlara konukçuluk yaparlar.',
            '**Dengeli Budama:** Bitkinin iç kısımlarının güneş almasını sağlayarak spor çimlenmesini engelleyin.',
            '**Doğal Fungisit:** Kekik veya okaliptüs yağlarının suyla seyreltilmiş çözeltileri mantar gelişimini baskılayabilir.'
        ]
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'name': 'Domates - Sarı Yaprak Kıvırcıklık Virüsü',
        'scientific_name': 'Begomovirus (TYLCV)',
        'severity': 'Kritik Tehdit',
        'severity_color': '#FF3333',
        'symptoms': 'Bitki gelişimi durur ve cüceleşir. Yeni çıkan yapraklar aşırı derecede yukarı doğru kıvrılır, küçülür ve kenarlarından sararır. Çiçek dökümü çok fazladır, meyve tutumu neredeyse sıfıra iner.',
        'organic_treatments': [
            '**Vektör Kontrolü (Beyazsinek):** Bu virüs beyazsinekler (Bemisia tabaci) ile taşınır. Beyazsinekleri izlemek ve yakalamak için bahçenin her yerine sarı yapışkan tuzaklar yerleştirin.',
            '**Enfekte Bitkiyi Sökün:** Virüsün tedavisi yoktur. Enfekte bitkiyi görür görmez sökün, poşete koyup bahçeden uzaklaştırın ve yakın.',
            '**Tül Örtü:** Küçük fideleri beyazsinek geçirmeyen mikro tüller ile koruyun.',
            '**Neem Yağı:** Beyazsinek yumurta ve larvalarını baskılamak için yaprak altlarına Neem yağı uygulayın.'
        ]
    },
    'Tomato__Tomato_mosaic_virus': {
        'name': 'Domates - Mozaik Virüsü',
        'scientific_name': 'Tobamovirus (ToMV)',
        'severity': 'Kritik Tehdit',
        'severity_color': '#FF3333',
        'symptoms': 'Yapraklarda açık ve koyu yeşil alanların karışımından oluşan mozaik desenleri görülür. Yapraklar büzüşür, incelir (eğrelti otu yaprağı gibi) ve bitki genel olarak zayıflar.',
        'organic_treatments': [
            '**Derhal İmha:** Virüslü bitkileri hemen kökleriyle birlikte sökün ve yakın (kesinlikle kompost yapmayın, virüs toprakta yıllarca yaşayabilir).',
            '**Süt Banyosu Alet Temizliği:** Bahçe aletlerinizi enfekte bitkiye dokunduktan sonra yağsız süte batırın. Sütteki proteinler mozaik virüsünü inaktive eder.',
            '**Tütün Teması Yasağı:** Mozaik virüsü tütünde de bulunur. Bahçede çalışmadan önce ellerinizi sabunla iyice yıkayın (sigara içiyorsanız çok dikkat edin).',
            '**Dayanıklı Hibrit:** Sonraki ekimlerde "ToMV" dayanıklılığı olan tohumlar tercih edin.'
        ]
    },
    'Tomato_healthy': {
        'name': 'Domates - Sağlıklı',
        'scientific_name': 'Solanum lycopersicum (Sağlıklı Bitki)',
        'severity': 'Sağlıklı',
        'severity_color': '#00cc66',
        'symptoms': 'Yapraklar canlı koyu yeşil, lekesiz ve diktir. Çiçeklenme ve meyve tutumu sağlıklı bir formda devam etmektedir.',
        'organic_treatments': [
            '**Kalsiyum Desteği:** Domateslerde çiçek burnu çürüklüğünü önlemek için toprağa yumurta kabuğu unu veya organik kalsiyum ekleyin.',
            '**Düzenli Budama:** Koltuk alma (gövde ile dal arasından çıkan gereksiz sürgünler) işlemini düzenli yaparak enerjiyi meyveye yönlendirin.',
            '**Malçlama:** Toprak nemini sabit tutmak için organik malç kullanın.'
        ]
    }
}

# --- STREAMLIT ARAYÜZÜ ---

# 1. Sayfa Ayarları (Premium Görünüm)
st.set_page_config(
    page_title="PlantGuard AI - Yaprak Hastalığı Teşhis Sistemi",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Özel CSS (Koyu Tema, Yumuşak Geçişler ve Glassmorphism)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Global Arayüz Özelleştirmeleri */
    * {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0d1b15 0%, #070f0b 100%);
        color: #e0e6e3;
    }
    
    /* Sidebar Tasarımı */
    section[data-testid="stSidebar"] {
        background-color: #0b1510 !important;
        border-right: 1px solid #1a3325;
    }
    
    /* Başlık ve Banner */
    .banner-container {
        background: linear-gradient(90deg, #102a1e 0%, #07140e 100%);
        border: 1px solid #1e4d37;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        gap: 20px;
    }
    
    .banner-title {
        color: #00ff88;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.3rem;
        margin: 0;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
    }
    
    .banner-subtitle {
        color: #a3b8ad;
        font-size: 1.1rem;
        margin: 5px 0 0 0;
    }
    
    /* Cam Efektli Kartlar (Glassmorphic Cards) */
    .glass-card {
        background: rgba(16, 35, 26, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(30, 77, 55, 0.3);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.3s ease, border 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(0, 255, 136, 0.4);
    }
    
    /* Sonuç Kartları */
    .result-badge {
        font-size: 1.6rem;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 10px;
        display: inline-block;
        margin-bottom: 15px;
    }
    
    .scientific-name {
        font-style: italic;
        color: #8c9e94;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00b368, #00ff88);
    }
    
    /* Sekme Tasarımı */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(16, 35, 26, 0.3);
        border: 1px solid rgba(30, 77, 55, 0.2);
        border-radius: 10px;
        color: #a3b8ad;
        padding: 0px 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0, 255, 136, 0.1);
        color: #00ff88;
        border-color: rgba(0, 255, 136, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 255, 136, 0.15) !important;
        color: #00ff88 !important;
        border-color: #00ff88 !important;
    }
    
    /* Custom Liste Elemanı */
    .treatment-item {
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #00ff88;
        padding: 12px 15px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Resim Yükleme Alanı */
    [data-testid="stFileUploader"] {
        background-color: rgba(16, 35, 26, 0.2);
        border: 2px dashed rgba(0, 255, 136, 0.3);
        border-radius: 15px;
        padding: 10px;
        transition: border 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #00ff88;
    }
    
    /* Tarama Animasyonu Efekti */
    .scanner-overlay {
        position: relative;
        overflow: hidden;
        border-radius: 12px;
        border: 2px solid #1e4d37;
    }
    
    .scanner-line {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(180deg, rgba(0,255,136,0) 0%, rgba(0,255,136,1) 50%, rgba(0,255,136,0) 100%);
        box-shadow: 0 0 15px #00ff88;
        animation: scan 2s linear infinite;
        z-index: 10;
    }
    
    @keyframes scan {
        0% { top: 0%; }
        50% { top: 100%; }
        100% { top: 0%; }
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Yükleme Fonksiyonu (Hızlı & Güvenli Cache Modu)
@st.cache_resource
def load_plant_model():
    model_name = 'plant_model_finetuned.keras'
    # Eğer fine-tune modeli yoksa ilk aşama modelini dene
    if not os.path.exists(model_name):
        model_name = 'plant_model_stage1.keras'
        
    if not os.path.exists(model_name):
        return None, f"Model dosyası bulunamadı! Lütfen '{model_name}' dosyasını bu klasöre kopyalayın."
    
    try:
        # Modeli compile=False ve safe_mode=False ile yükleyip yapısını okuyoruz
        loaded_model = None
        try:
            loaded_model = tf.keras.models.load_model(model_name, compile=False, safe_mode=False)
        except TypeError:
            # safe_mode desteklemeyen eski Keras/TF sürümleri için
            loaded_model = tf.keras.models.load_model(model_name, compile=False)
            
        if loaded_model is None:
            return None, "Model dosyası yüklenirken okunamadı."
            
        # Sadece tahmin için gereken ağırlıklı katmanları (xception, pooling, dropout, dense) ayırıyoruz.
        # Böylece hata veren Lambda (keras.applications...) ve RandomFlip katmanlarını baypas ediyoruz.
        
        # 1. Giriş katmanı tanımlıyoruz (224x224x3)
        input_layer = tf.keras.layers.Input(shape=(224, 224, 3), name="input_clean")
        
        # 2. Modeli tarayarak Xception ve sonrasındaki katmanları buluyoruz
        x = input_layer
        started = False
        
        for layer in loaded_model.layers:
            # Xception katmanını ve sonrasını bağlıyoruz
            if 'xception' in layer.name or isinstance(layer, tf.keras.Model) or layer.__class__.__name__ == 'Functional':
                started = True
            
            if started:
                x = layer(x)
                
        if not started:
            # Eğer xception katmanı ismen bulunamazsa, ilk 3 katmanı (Flip, Rotation, Lambda) atlayıp 
            # geri kalan katmanları bağlıyoruz.
            for layer in loaded_model.layers[3:]:
                x = layer(x)
                
        # 3. Yeni temiz modeli oluşturuyoruz
        reconstructed_model = tf.keras.models.Model(inputs=input_layer, outputs=x)
        return reconstructed_model, model_name
        
    except Exception as err:
        return None, f"Model yüklenirken ve yeniden yapılandırılırken hata oluştu: {str(err)}"

# Modeli arka planda yükle
model, model_status = load_plant_model()

# 4. Sol Kenar Çubuğu (Sidebar)
with st.sidebar:
    st.markdown("<h2 style='color:#00ff88; font-family:Outfit; margin-bottom:0;'>🍃 PlantGuard AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8c9e94; font-size:0.85rem; margin-top:0;'>Akıllı Yaprak Teşhis Paneli v1.2</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Model Bilgisi ve Sağlık Durumu
    st.markdown("### 📊 Model Durumu")
    if model is not None:
        st.success(f"✓ Model Hazır: \n`{model_status}`")
        
        # Model Özellikleri Tablosu
        st.markdown(
            f"""
            <table style='width:100%; font-size:0.85rem; color:#a3b8ad;'>
                <tr><td><b>Mimari:</b></td><td>Xception DeepNet</td></tr>
                <tr><td><b>Çözünürlük:</b></td><td>224 x 224 x 3</td></tr>
                <tr><td><b>Toplam Sınıf:</b></td><td>{len(CLASS_NAMES)} Kategori</td></tr>
                <tr><td><b>Platform:</b></td><td>TensorFlow & Keras</td></tr>
            </table>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.error(f"❌ Model Yüklenemedi!")
        st.info(model_status)
        
    st.markdown("---")
    
    # Colab Eğitim Geçmişi (Özet Bilgi)
    st.markdown("### 📈 Colab Eğitim Özeti")
    st.markdown(
        """
        - **1. Aşama (Transfer Learning):**
          - Epoch: 0 - 20 (Dondurulmuş Xception)
          - Öğrenme Oranı (LR): 1e-3 (Adam)
        - **2. Aşama (Fine-Tuning):**
          - Epoch: 20 - 35 (Çözülmüş Katmanlar)
          - Öğrenme Oranı (LR): 1e-5 (Adam)
        - **Görüntü Artırma:** Horizontal/Vertical Flip, 0.2 Random Rotation.
        """
    )
    


# 5. Ana Ekran - Banner
st.markdown(
    """
    <div class="banner-container">
        <div style="font-size: 3rem; filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.4));">🌿</div>
        <div>
            <h1 class="banner-title">PlantGuard AI</h1>
            <p class="banner-subtitle">Xception Derin Öğrenme Tabanlı Yaprak Hastalığı Teşhis & Organik Tedavi Portalı</p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# 6. Model Yükleme Başarısız Uyarısı
if model is None:
    st.error("⚠️ Lütfen dikkat! Model dosyası bulunamadı veya yüklenemedi.")
    st.warning("Uygulamanın çalışabilmesi için Colab'da eğittiğiniz **`plant_model_finetuned.keras`** dosyasını bu script ile aynı klasöre taşımalısınız.")
    st.stop()

# 7. Resim Yükleme Aşaması
st.markdown("### 📸 Bitki Yaprağı Fotoğrafı Yükleyin")
uploaded_file = st.file_uploader(
    "Teşhis etmek istediğiniz Biber, Patates veya Domates yaprağının fotoğrafını seçin (PNG, JPG, JPEG)...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # İki sütunlu düzen: Sol sütun Resim & Sağ sütun Sonuçlar
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#00ff88; margin-top:0;'>Yüklenen Yaprak Görseli</h4>", unsafe_allow_html=True)
        
        # PIL ile resmi aç
        image = Image.open(uploaded_file).convert('RGB')
        
        # Premium Tarama Animasyonlu Resim Çerçevesi
        st.markdown(
            """
            <div class="scanner-overlay">
                <div class="scanner-line"></div>
            """, 
            unsafe_allow_html=True
        )
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#00ff88; margin-top:0;'>Teşhis ve Analiz Sonuçları</h4>", unsafe_allow_html=True)
        
        # Yapay Zeka Analiz Progress/Loading efekti
        with st.spinner("🧠 Yapay zeka modeli yaprağı analiz ediyor..."):
            # Resmi modele uygun şekle getir
            # 1. 224x224 boyutuna küçült
            resized_image = image.resize((224, 224))
            # 2. Numpy dizisine çevir (float32)
            img_array = np.array(resized_image, dtype=np.float32)
            # 3. Xception için manuel ön işleme (0-255 arasını -1.0 ile 1.0 arasına ölçekler)
            # Baypas edilen Lambda katmanının görevini burada kod içinde yapıyoruz.
            img_array = (img_array / 127.5) - 1.0
            # 4. Batch boyutu ekle (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Tahmin işlemi (Baypas edilmiş temiz model ile, terminal log kalabalığını önlemek için verbose=0)
            predictions = model.predict(img_array, verbose=0)
            # En yüksek olasılığa sahip sınıfı ve olasılık değerini al
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Gecikme simülasyonu (premium UX hissi için)
            time.sleep(0.8)
            
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        
        # Hastalık Bilgilerini Veritabanından Al
        disease_info = PLANT_DISEASES_DB.get(
            predicted_class_name, 
            {
                'name': predicted_class_name,
                'scientific_name': 'Bilinmeyen Bitki Türü',
                'severity': 'Bilinmiyor',
                'severity_color': '#a3b8ad',
                'symptoms': 'Bu hastalık için semptom bilgisi bulunamadı.',
                'organic_treatments': ['Genel bitki bakımı uygulayın.', 'Aşırı sulamadan kaçının.']
            }
        )
        
        # Sonucu Ekrana Yazdır
        color = disease_info['severity_color']
        st.markdown(
            f"""
            <div style='margin-bottom: 5px; color:#8c9e94; font-size:0.9rem; font-weight:600;'>BULGULANAN TEŞHİS</div>
            <div style='color: {color}; font-size: 2.1rem; font-weight: 700; line-height:1.2; margin-bottom:5px;'>
                {disease_info['name']}
            </div>
            <div class="scientific-name">Bilimsel Adı: {disease_info['scientific_name']}</div>
            """, 
            unsafe_allow_html=True
        )
        
        # Güven Skoru ve Risk Durumu Yanyana
        c_val1, c_val2 = st.columns(2)
        with c_val1:
            st.markdown(
                f"""
                <div style='background:rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding:10px 15px; border-radius:8px;'>
                    <span style='color:#8c9e94; font-size:0.8rem; display:block;'>GÜVEN SKORU</span>
                    <span style='color:#00ff88; font-size:1.6rem; font-weight:700;'>%{confidence*100:.2f}</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with c_val2:
            st.markdown(
                f"""
                <div style='background:rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding:10px 15px; border-radius:8px;'>
                    <span style='color:#8c9e94; font-size:0.8rem; display:block;'>TEHDİT SEVİYESİ</span>
                    <span style='color:{color}; font-size:1.6rem; font-weight:700;'>{disease_info['severity']}</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; color:#8c9e94; margin-bottom:5px;'>Teşhis Olasılığı</p>", unsafe_allow_html=True)
        st.progress(float(confidence))
        
        st.markdown("</div>", unsafe_allow_html=True) # glass-card sonu

    # Sekmeli Detay Paneli (Teşhis, Tedavi, Olasılık Grafiği)
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs([
        "📋 Detaylı Teşhis Raporu", 
        "🛡️ Organik Tedavi & Bakım Rehberi", 
        "📊 Tüm Kategori Olasılıkları"
    ])
    
    with tab1:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="color:#00ff88; margin-top:0; border-bottom:1px solid rgba(0, 255, 136, 0.2); padding-bottom:10px;">🩺 Hastalık Belirtileri ve Patolojik Bulgular</h4>
                <p style="font-size: 1.05rem; line-height: 1.6; color: #e0e6e3; margin-top:15px;">
                    {disease_info['symptoms']}
                </p>
                <div style="background-color: rgba(255, 153, 0, 0.05); border: 1px solid rgba(255, 153, 0, 0.2); padding: 15px; border-radius: 8px; margin-top: 20px;">
                    <span style="color:#FF9900; font-weight:700; font-size:1.1rem; display:block; margin-bottom:5px;">⚠️ Önemli Hatırlatma</span>
                    <span style="color:#d4dbd7; font-size:0.92rem;">
                        Bu analiz yapay zeka tarafından yapılmıştır. Tarımsal verimliliğinizi korumak amacıyla büyük alanlarda işlem yapmadan önce yerel tarım danışmanınızdan veya il/ilçe tarım müdürlüğünden teyit almanız tavsiye edilir.
                    </span>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            """
            <h4 style="color:#00ff88; margin-top:0; border-bottom:1px solid rgba(0, 255, 136, 0.2); padding-bottom:10px;">
                🌿 Çevre Dostu ve Doğal Çözüm Reçeteleri
            </h4>
            <p style="color:#a3b8ad; font-size:0.95rem; margin-top:10px; margin-bottom:20px;">
                Kimyasal kalıntı bırakmayan, toprağı ve faydalı böcekleri koruyan organik mücadele yöntemleri:
            </p>
            """, 
            unsafe_allow_html=True
        )
        
        for treatment in disease_info['organic_treatments']:
            st.markdown(
                f"""
                <div class="treatment-item">
                    {treatment}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#00ff88; margin-top:0;'>📊 Tüm Olasılık Dağılımları (En Yüksek 5 Sınıf)</h4>", unsafe_allow_html=True)
        
        # En yüksek 5 olasılığı sırala
        top_indices = np.argsort(predictions[0])[::-1][:5]
        top_classes = [CLASS_NAMES[i] for i in top_indices]
        top_scores = [predictions[0][i] for i in top_indices]
        
        # Türkçe sınıf isimlerini eşle
        top_turkish_names = []
        for c in top_classes:
            top_turkish_names.append(PLANT_DISEASES_DB.get(c, {'name': c})['name'])
            
        # Matplotlib ile şık yatay grafik oluştur
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4.5))
        
        # Renk paleti (Gradient hissiyatı)
        colors = ['#00ff88', '#00e676', '#66bb6a', '#81c784', '#a5d6a7']
        
        bars = ax.barh(top_turkish_names[::-1], np.array(top_scores[::-1]) * 100, color=colors[::-1], height=0.5)
        
        # Kenarlıkları kaldır
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_color('#1e4d37')
        
        ax.xaxis.grid(True, linestyle='--', alpha=0.2, color='#00ff88')
        ax.set_xlabel('Güven Yüzdesi (%)', color='#a3b8ad', fontsize=10, labelpad=10)
        
        # Grafik içi yüzdelik etiketleri ekle
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'%{width:.1f}', 
                    va='center', ha='left', color='#e0e6e3', fontweight='semibold', fontsize=10)
            
        fig.patch.set_facecolor('#0d1b15')
        ax.set_facecolor('#0d1b15')
        
        # Grafik etiketlerinin yazı boyutunu ayarla
        ax.tick_params(colors='#a3b8ad', labelsize=11)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Görsel yüklenmediğinde şık bir rehber göster
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 50px 20px;">
            <div style="font-size: 4.5rem; margin-bottom: 20px;">📸</div>
            <h3 style="color:#00ff88; margin-top:0;">Yaprak Fotoğrafı Bekleniyor</h3>
            <p style="color:#a3b8ad; max-width: 600px; margin: 0 auto; line-height: 1.6; font-size:1.05rem;">
                Bilgisayarınızdan veya telefonunuzdan teşhis edilmesini istediğiniz bitki yaprağının yakın plan fotoğrafını yükleyin. 
                Sistemimiz <b>Domates</b>, <b>Patates</b> ve <b>Biber</b> yapraklarındaki 15 farklı sağlık ve hastalık durumunu teşhis edebilir.
            </p>
            <div style="margin-top: 30px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
                <span style="background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.3); color: #00ff88; padding: 6px 16px; border-radius: 20px; font-size: 0.9rem; font-weight:600;">🍅 Domates (9 Hastalık + Sağlıklı)</span>
                <span style="background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.3); color: #00ff88; padding: 6px 16px; border-radius: 20px; font-size: 0.9rem; font-weight:600;">🥔 Patates (2 Hastalık + Sağlıklı)</span>
                <span style="background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.3); color: #00ff88; padding: 6px 16px; border-radius: 20px; font-size: 0.9rem; font-weight:600;">🫑 Biber (1 Hastalık + Sağlıklı)</span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# 8. Footer (Alt Bilgi)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #5a6e63; font-size: 0.85rem; border-top: 1px solid rgba(30, 77, 55, 0.2); padding-top: 20px; margin-bottom: 20px;">
        PlantGuard AI © 2026. Tüm hakları saklıdır. <br>
        <i>Deep Learning Model: Xception Transfer Learning & Fine-Tuned (35 Epochs)</i>
    </div>
    """, 
    unsafe_allow_html=True
)

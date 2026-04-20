# Görüntü İşleme Dersi — Grup 12

## Temel Görüntü İşleme Operasyonlarının Manuel İmplementasyonu

Bu projede temel görüntü işleme operasyonları, hazır kütüphane fonksiyonları kullanılmaksızın sıfırdan (from scratch) manuel olarak implement edilmektedir.

---

## Grup Üyeleri ve İş Bölümü

| Kişi | Sorumluluk | Dosya |
|------|-----------|-------|
| Kişi 1 | Gri Dönüşüm, Binary, Renk Uzayı (RGB→HSV), Histogram | `kisi1_temel.py` |
| Kişi 2 | Döndürme, Kırpma, Ölçekleme, Aritmetik İşlemler | `kisi2_geometrik.py` |
| Kişi 3 | Parlaklık/Kontrast, Konvolüsyon, Gauss, Bulanıklaştırma | `kisi3_filtreleme.py` |
| Kişi 4 | Eşikleme (Global & Adaptif), Sobel Kenar, Gürültü | `kisi4_kenar.py` |
| Kişi 5 | Morfolojik İşlemler, Ana İş Akışı, Entegrasyon, Arayüz | `kisi5_morfoloji.py` + `arayuz.py` |

---

## Kullanılan Teknolojiler

- **Python 3.10+** — ana geliştirme dili
- **NumPy** — tüm piksel işlemleri için (serbest)
- **OpenCV** — yalnızca `cv2.imread`, `cv2.imwrite`, `cv2.imdecode` (sınırlı)
- **Matplotlib** — görselleştirme ve histogram grafikleri
- **Pillow** — arayüz görsel desteği (ImageTk)

---

## Proje Yapısı

```
goruntu-isleme-proje/
├── images/                  ← test görüntüleri (buraya koy)
├── outputs/                 ← işlenmiş çıktılar (.gitignore'da)
├── arayuz.py                ← ANA ARAYÜZ (Buradan çalıştırın)
├── kisi1_temel.py
├── kisi2_geometrik.py
├── kisi3_filtreleme.py
├── kisi4_kenar.py
├── kisi5_morfoloji.py
├── main.py                  ← terminal tabanlı ana akış
├── test_goruntu_olustur.py  ← hızlı test için görsel üretici
├── README.md
└── .gitignore
```

---

## Kurulum

```bash
pip install numpy opencv-python matplotlib pillow
```

---

## Çalıştırma

```bash
python arayuz.py
```

Terminal akışı için: `python main.py`

---

## Geliştiriciler İçin Önemli Notlar

### Kendi Modülünüzü Arayüze Ekleme

1. `arayuz.py` dosyasının en üstünde kendi dosyanızı `import` edin.
2. `KisiXSekmesi` sınıfı içindeki placeholder kısmını kaldırıp kendi buton ve slider'larınızı ekleyin.
3. Ağır işlemlerin arayüzü dondurmaması için Kişi 5'in yazdığı Threading yapısını örnek alın.

### Türkçe Karakter ve Dosya Yolu Sorunu

Dosya yolunda Türkçe karakter (`ü, İ, ş, ğ` vb.) varsa OpenCV hata verebilir. Bu proje `np.fromfile` + `cv2.imdecode` kullanarak bu sorunu çözmüştür. Kendi dosya işlemlerinizde `arayuz.py` içindeki `_goruntu_yukle` metodunu referans alın.

---

## Sıkça Karşılaşılan Hatalar ve Çözümleri

### "ModuleNotFoundError: No module named 'cv2'"

OpenCV yüklü değil. Terminale şunu yaz:

```bash
pip install numpy opencv-python matplotlib pillow
```

Birden fazla Python sürümü varsa:

```bash
python -m pip install numpy opencv-python matplotlib pillow
```

---

### "Goruntu dosyasi bulunamadi: 'images/test.jpg'"

Henüz test görseli oluşturulmamış. Şunu çalıştır:

```bash
python test_goruntu_olustur.py
```

Sonra tekrar `python arayuz.py` de.

---

### Arayüz açılıyor ama görüntü yüklenmiyor

Dosya yolunda Türkçe karakter var mı kontrol et. Görseli `images/` klasörüne taşı ve oradan aç. Klasörün adında da Türkçe karakter olmamasına dikkat et.

---

### "pip" komutu tanınmıyor (Windows)

Python PATH'e eklenmemiş. Şunu dene:

```bash
py -m pip install numpy opencv-python matplotlib pillow
```

Hâlâ çalışmıyorsa Python'u kaldırıp yeniden kur, kurulum sırasında **"Add Python to PATH"** kutucuğunu işaretle.

---

### Kod çalışıyor ama çıktı görüntüsü kaydedilmiyor

`outputs/` klasörünün var olduğundan emin ol. Yoksa oluştur:

```bash
mkdir outputs
```

---

## Branch Kuralları

- Her kişi kendi branch'inde çalışır: `kisi1/temel`, `kisi2/geometrik` vb.
- `main` branch'e doğrudan push yapılmaz — Pull Request (PR) açılır.
- PR açmadan önce kendi modülünü izole test et.

---

## Kaynaklar

1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Yildirim, A., Kose, C., & Sengur, A. (2021). Effect of color space transformations on segmentation performance in medical images. *Biomedical Signal Processing and Control*, 65, 102359.
3. Bradley, D., & Roth, G. (2007). Adaptive image thresholding using the integral image. *Journal of Graphics Tools*, 12(2), 13–21.
4. Lindeberg, T. (1994). Scale-space theory: A basic tool for analyzing structures at different scales. *Journal of Applied Statistics*, 21(1-2), 225–270.
5. Huang, T. S., Yang, G. J., & Tang, G. Y. (1979). A fast two-dimensional median filtering algorithm. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 27(1), 13–18.
6. Sobel, I., & Feldman, G. (1968). A 3x3 isotropic gradient operator for image processing. Stanford AI Project.
7. Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.

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
| Kişi 5 | Morfolojik İşlemler, Ana İş Akışı, Entegrasyon | `kisi5_morfoloji.py` + `main.py` |

---

## Kullanılan Teknolojiler

- **Python 3.10+** — ana geliştirme dili
- **NumPy** — tüm piksel işlemleri için (serbest)
- **OpenCV** — yalnızca `cv2.imread`, `cv2.imwrite`, `cv2.imshow` (sınırlı)
- **Matplotlib** — görselleştirme ve histogram grafikleri

### Yasak Fonksiyonlar

OpenCV'nin şu fonksiyonlarının kullanımı kesinlikle yasaktır:
`cv2.resize`, `cv2.rotate`, `cv2.warpAffine`, `cv2.filter2D`, `cv2.GaussianBlur`,
`cv2.medianBlur`, `cv2.Sobel`, `cv2.Canny`, `cv2.threshold`, `cv2.adaptiveThreshold`,
`cv2.equalizeHist`, `cv2.erode`, `cv2.dilate`

---

## Proje Yapısı

```
goruntu-isleme-proje/
├── images/              ← test görüntüleri (buraya koy)
├── outputs/             ← işlenmiş çıktılar (.gitignore'da)
├── kisi1_temel.py
├── kisi2_geometrik.py
├── kisi3_filtreleme.py
├── kisi4_kenar.py
├── kisi5_morfoloji.py
├── main.py              ← tüm modülleri çalıştırır
├── README.md
└── .gitignore
```

---

## Kurulum

```bash
pip install numpy opencv-python matplotlib
```

---

## Çalıştırma

```bash
python main.py
```

---

## Branch Kuralları

- Her kişi kendi branch'inde çalışır: `kisi1/temel`, `kisi2/geometrik` vb.
- `main` branch'e doğrudan push yapılmaz — Pull Request açılır.
- PR açmadan önce kendi modülünü izole test et.

---

## Kapsanan Modüller

1. Gri tonlama dönüşümü (ITU-R BT.601)
2. Binary dönüşüm
3. Görüntü döndürme (inverse mapping)
4. Görüntü kırpma
5. Yakınlaştırma/uzaklaştırma (Nearest Neighbor)
6. Renk uzayı dönüşümü (RGB → HSV)
7. Histogram hesabı ve germe
8. Aritmetik işlemler (toplama, çarpma)
9. Parlaklık ve kontrast ayarı
10. Konvolüsyon ve Gauss filtresi
11. Eşikleme (global ve adaptif)
12. Sobel kenar bulma operatörü
13. Gürültü ekleme (Salt & Pepper) ve temizleme
14. Bulanıklaştırma filtreleri (Mean, Gauss)
15. Morfolojik işlemler (Dilation, Erosion, Opening, Closing)

---

## Kaynaklar

1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Yildirim, A., Kose, C., & Sengur, A. (2021). Effect of color space transformations on segmentation performance in medical images. *Biomedical Signal Processing and Control*, 65, 102359.
3. Bradley, D., & Roth, G. (2007). Adaptive image thresholding using the integral image. *Journal of Graphics Tools*, 12(2), 13–21.
4. Lindeberg, T. (1994). Scale-space theory: A basic tool for analyzing structures at different scales. *Journal of Applied Statistics*, 21(1-2), 225–270.
5. Huang, T. S., Yang, G. J., & Tang, G. Y. (1979). A fast two-dimensional median filtering algorithm. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 27(1), 13–18.
6. Sobel, I., & Feldman, G. (1968). A 3x3 isotropic gradient operator for image processing. Stanford AI Project.
7. Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.

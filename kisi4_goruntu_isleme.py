"""
=============================================================
KİŞİ 4 - GÖRÜNTÜ İŞLEME PROJESİ
=============================================================
İçerik:
  4.11  Global & Adaptif Eşikleme
  4.12  Sobel Kenar Bulma
  4.13  Salt & Pepper Gürültü Ekleme + Mean & Median Filtre ile Temizleme
  4.14  Bulanıklaştırma Filtreleri (Mean Blur, Gaussian Blur)

Kurallar:
  ✅  numpy, cv2.imread / cv2.imwrite / cv2.imshow, matplotlib
  ❌  cv2.threshold, cv2.adaptiveThreshold, cv2.Sobel, cv2.filter2D,
      cv2.blur, cv2.GaussianBlur, cv2.medianBlur  → YASAK
=============================================================
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


# ─────────────────────────────────────────────────────────────
# YARDIMCI: Gri dönüşüm
# ─────────────────────────────────────────────────────────────
def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    BGR görüntüyü gri tonlamaya dönüştürür.
    Formül (ITU-R BT.601): Gray = 0.299*R + 0.587*G + 0.114*B
    OpenCV kanalları B-G-R sırasındadır.
    """
    b = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    r = image[:, :, 2].astype(np.float64)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# YARDIMCI: Manuel 2-D konvolüsyon (zero-padding)
# ─────────────────────────────────────────────────────────────
def manuel_konvolusyon(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Gri görüntüyü verilen kernel ile konvolüsyona sokar.
    - Zero-padding uygulanır (kenar efekti minimum).
    - Hazır filtre fonksiyonu (cv2.filter2D vb.) KULLANILMAZ.

    Parametreler
    ------------
    gray   : 2-D uint8 dizi (H x W)
    kernel : 2-D float64 dizi (kH x kW)  — kH ve kW tek sayı olmalı

    Dönüş
    -----
    output : 2-D uint8 dizi, [0-255] aralığında kırpılmış
    """
    H, W = gray.shape
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2

    # Zero-padding
    padded = np.zeros((H + 2 * pad_h, W + 2 * pad_w), dtype=np.float64)
    padded[pad_h:pad_h + H, pad_w:pad_w + W] = gray.astype(np.float64)

    output = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            bolge = padded[i:i + kH, j:j + kW]    # kernel boyutunda komşuluk
            output[i, j] = np.sum(bolge * kernel)  # eleman bazlı çarpım + toplam

    return np.clip(output, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# 4.11  GLOBAL & ADAPTİF EŞİKLEME
# ─────────────────────────────────────────────────────────────
def global_esikleme(gray: np.ndarray, esik: int = 127) -> np.ndarray:
    """
    Tüm görüntüye tek bir eşik değeri uygular.
    binary[x,y] = 255 eğer gray[x,y] >= esik, aksi hâlde 0

    np.where vektörize yapısı kullanılır; hazır cv2.threshold YASAK.
    """
    binary = np.where(gray >= esik, 255, 0).astype(np.uint8)
    return binary


def adaptif_esikleme(gray: np.ndarray, pencere_boyutu: int = 11, C: int = 5) -> np.ndarray:
    """
    Her piksel için eşik değerini yerel komşuluk ortalamasından hesaplar.

    Algoritma (Bradley & Roth, 2007 esas alınarak):
      T(x,y) = mean( komşuluk(x,y) ) - C

    Adımlar:
      1. Görüntüye zero-padding uygulanır (kenar pikselleri için).
      2. Her (i,j) pikseli için etrafındaki pencere_boyutu x pencere_boyutu blok alınır.
      3. Bloğun aritmetik ortalaması hesaplanır: T = ortalama - C
      4. gray[i,j] >= T ise piksel 255, değilse 0 yapılır.

    cv2.adaptiveThreshold YASAK; tamamen manuel döngülerle implement edilir.

    Parametreler
    ------------
    gray           : 2-D uint8 gri görüntü
    pencere_boyutu : Yerel komşuluk boyutu (tek sayı, örn. 11)
    C              : Ortalamadan çıkarılacak sabit
    """
    assert pencere_boyutu % 2 == 1, "pencere_boyutu tek sayı olmalıdır!"

    H, W = gray.shape
    pad = pencere_boyutu // 2
    binary = np.zeros((H, W), dtype=np.uint8)

    # Zero-padding
    padded = np.zeros((H + 2 * pad, W + 2 * pad), dtype=np.float64)
    padded[pad:pad + H, pad:pad + W] = gray.astype(np.float64)

    for i in range(H):
        for j in range(W):
            blok = padded[i:i + pencere_boyutu, j:j + pencere_boyutu]
            yerel_ortalama = np.mean(blok)       # NumPy ortalaması
            esik_degeri = yerel_ortalama - C     # T(x,y) = mean - C
            binary[i, j] = 255 if gray[i, j] >= esik_degeri else 0

    return binary


# ─────────────────────────────────────────────────────────────
# 4.12  SOBEL KENAR BULMA
# ─────────────────────────────────────────────────────────────
def sobel_kenar_bulma(gray: np.ndarray, esik: int = 50):
    """
    Sobel operatörüyle kenar tespiti.

    Maskeler (Sobel & Feldman, 1968):
      Gx = [[-1, 0, 1],        Gy = [[-1, -2, -1],
             [-2, 0, 2],               [ 0,  0,  0],
             [-1, 0, 1]]               [ 1,  2,  1]]

    Gradyan büyüklüğü : G  = sqrt(Gx^2 + Gy^2)
    Gradyan yönü      : theta = atan2(Gy, Gx)  [radyan]
    Kenar görüntüsü   : edges[x,y] = 255 eğer G[x,y] >= esik

    cv2.Sobel ve cv2.filter2D KULLANILMAZ.

    Dönüş
    -----
    edges     : 2-D uint8 kenar görüntüsü
    magnitude : 2-D float64 gradyan büyüklüğü
    direction : 2-D float64 gradyan yönü (radyan)
    """
    # Sobel maskeleri
    Gx_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)

    Gy_kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float64)

    # Her iki yön için ayrı konvolüsyon (manuel)
    Gx = manuel_konvolusyon(gray, Gx_kernel).astype(np.float64)
    Gy = manuel_konvolusyon(gray, Gy_kernel).astype(np.float64)

    # Gradyan büyüklüğü: G = sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

    # Gradyan yönü: theta = atan2(Gy, Gx)
    direction = np.arctan2(Gy, Gx)

    # Eşikleme ile kenar maskesi
    edges = np.where(magnitude >= esik, 255, 0).astype(np.uint8)

    return edges, magnitude, direction


# ─────────────────────────────────────────────────────────────
# 4.13  SALT & PEPPER GURULTU + MEAN & MEDIAN FILTRE
# ─────────────────────────────────────────────────────────────
def salt_pepper_gurultu_ekle(gray: np.ndarray, gurultu_orani: float = 0.05) -> np.ndarray:
    """
    Görüntüye Salt & Pepper (Tuz-Biber) gürültüsü ekler.

    Algoritma:
      1. [0,1) aralığında görüntüyle aynı boyutta rastgele matris üretilir.
      2. prob < gurultu_orani/2        → piksel = 0   (pepper/biber)
         prob > 1 - gurultu_orani/2   → piksel = 255 (salt/tuz)
         diğerleri                    → orijinal değer korunur.
    """
    gurultulu = gray.copy()
    prob = np.random.random(gray.shape)

    gurultulu[prob < gurultu_orani / 2] = 0           # pepper
    gurultulu[prob > 1 - gurultu_orani / 2] = 255     # salt

    return gurultulu


def mean_filtre(gray: np.ndarray, pencere_boyutu: int = 3) -> np.ndarray:
    """
    Ortalama (Mean) Filtre — gürültü temizleme.

    Tüm elemanları 1/(k*k) olan kutu kerneli ile manuel konvolüsyon uygulanır.
    cv2.blur veya cv2.filter2D KULLANILMAZ.
    """
    k = pencere_boyutu
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    return manuel_konvolusyon(gray, kernel)


def median_filtre(gray: np.ndarray, pencere_boyutu: int = 3) -> np.ndarray:
    """
    Medyan (Median) Filtre — Salt & Pepper gürültüsüne karşı üstün.

    Medyan filtre kenarlara zarar vermeden dürtüsel gürültüyü temizler
    (Huang ve ark., 1979).

    Algoritma:
      Her piksel için pencere_boyutu x pencere_boyutu komşuluk alınır,
      değerlerin medyanı merkez piksele yazılır.

    np.median() kullanılır; cv2.medianBlur KULLANILMAZ.
    """
    H, W = gray.shape
    pad = pencere_boyutu // 2
    output = np.zeros((H, W), dtype=np.uint8)

    # Zero-padding
    padded = np.zeros((H + 2 * pad, W + 2 * pad), dtype=np.uint8)
    padded[pad:pad + H, pad:pad + W] = gray

    for i in range(H):
        for j in range(W):
            bolge = padded[i:i + pencere_boyutu, j:j + pencere_boyutu]
            output[i, j] = int(np.median(bolge))  # medyan değer

    return output


# ─────────────────────────────────────────────────────────────
# 4.14  BULANIKLAŞTIRMA FİLTRELERİ
# ─────────────────────────────────────────────────────────────
def mean_blur(gray: np.ndarray, pencere_boyutu: int = 5) -> np.ndarray:
    """
    Ortalama (Box) Bulanıklaştırma.

    Tüm komşu piksellere eşit ağırlık (1/k^2) verilir.
    cv2.blur veya cv2.filter2D KULLANILMAZ.
    """
    k = pencere_boyutu
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    return manuel_konvolusyon(gray, kernel)


def gaussian_kernel_olustur(boyut: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    2-D Gauss kerneli sıfırdan üretir.

    Formül (Lindeberg, 1994):
      G(x,y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))

    Kernel, toplamının 1 olması için normalizasyona tabi tutulur.
    """
    assert boyut % 2 == 1, "Kernel boyutu tek sayı olmalıdır!"
    merkez = boyut // 2
    kernel = np.zeros((boyut, boyut), dtype=np.float64)

    for x in range(boyut):
        for y in range(boyut):
            dx = x - merkez
            dy = y - merkez
            kernel[x, y] = (
                (1.0 / (2.0 * math.pi * sigma ** 2)) *
                math.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
            )

    kernel /= kernel.sum()  # normalizasyon
    return kernel


def gaussian_blur(gray: np.ndarray, kernel_boyutu: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Gauss Bulanıklaştırma.

    Merkeze yakın piksellere daha büyük ağırlık verilir; kenar bilgisi
    ortalama filtreden daha iyi korunur.
    cv2.GaussianBlur veya cv2.filter2D KULLANILMAZ.
    """
    kernel = gaussian_kernel_olustur(boyut=kernel_boyutu, sigma=sigma)
    return manuel_konvolusyon(gray, kernel)


# ─────────────────────────────────────────────────────────────
# DEMO & GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
def demo(goruntu_yolu: str):
    """
    Tüm modülleri çalıştırır, sonuçları kaydeder ve görselleştirir.
    """
    img_bgr = cv2.imread(goruntu_yolu)
    if img_bgr is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {goruntu_yolu}")

    gray = rgb_to_gray(img_bgr)

    # 4.11
    global_bin   = global_esikleme(gray, esik=127)
    adaptif_bin  = adaptif_esikleme(gray, pencere_boyutu=11, C=5)

    # 4.12
    edges, magnitude, direction = sobel_kenar_bulma(gray, esik=50)

    # 4.13
    gurultulu    = salt_pepper_gurultu_ekle(gray, gurultu_orani=0.05)
    mean_temiz   = mean_filtre(gurultulu, pencere_boyutu=3)
    median_temiz = median_filtre(gurultulu, pencere_boyutu=3)

    # 4.14
    mean_b  = mean_blur(gray, pencere_boyutu=5)
    gauss_b = gaussian_blur(gray, kernel_boyutu=5, sigma=1.4)

    # Kaydetme
    cv2.imwrite("cikti_global_esikleme.png",  global_bin)
    cv2.imwrite("cikti_adaptif_esikleme.png", adaptif_bin)
    cv2.imwrite("cikti_sobel_kenarlar.png",   edges)
    cv2.imwrite("cikti_salt_pepper.png",      gurultulu)
    cv2.imwrite("cikti_mean_filtre.png",      mean_temiz)
    cv2.imwrite("cikti_median_filtre.png",    median_temiz)
    cv2.imwrite("cikti_mean_blur.png",        mean_b)
    cv2.imwrite("cikti_gaussian_blur.png",    gauss_b)
    print("Tüm çıktı görüntüleri kaydedildi.")

    # Görselleştirme
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle("Kisi 4 - Goruntu Isleme Sonuclari", fontsize=16, fontweight="bold")

    goster = [
        (gray,          "Gri Goruntu",            "gray"),
        (global_bin,    "Global Esikleme",         "gray"),
        (adaptif_bin,   "Adaptif Esikleme",        "gray"),
        (magnitude,     "Sobel Gradyan Buyuklugu", "hot"),
        (edges,         "Sobel Kenarlar",           "gray"),
        (gurultulu,     "S&P Gurultu (%5)",         "gray"),
        (mean_temiz,    "Mean Filtre (3x3)",        "gray"),
        (median_temiz,  "Median Filtre (3x3)",      "gray"),
        (mean_b,        "Mean Blur (5x5)",           "gray"),
        (gauss_b,       "Gaussian Blur (sigma=1.4)","gray"),
        (img_bgr[:,:,::-1], "Orijinal RGB",          None),
        (direction,     "Sobel Gradyan Yonu",       "hsv"),
    ]

    for ax, (img, baslik, cmap) in zip(axes.flat, goster):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(baslik, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("kisi4_sonuclar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Gorsellestirme tamamlandi -> kisi4_sonuclar.png")


if __name__ == "__main__":
    # Test için proje klasöründeki bir görseli kullanıyoruz
    test_yolu = "images/test_photo.png"
    
    import os
    if os.path.exists(test_yolu):
        demo(test_yolu)
    else:
        print(f"Uyari: Test gorseli bulunamadi ({test_yolu}).")
        print("Lutfen arayuz.py uzerinden calistirin veya gecerli bir yol verin.")

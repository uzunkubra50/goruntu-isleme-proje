"""
kisi1_temel_donusumler.py
=========================
Kişi 1 - Temel Görüntü İşleme Dönüşümleri

Görevler:
    4.1 - Gri Tonlama Dönüşümü
    4.2 - Binary (İkili) Dönüşüm
    4.6 - Renk Uzayı Dönüşümü (BGR -> HSV)
    4.7 - Histogram Hesaplama ve Histogram Germe (Linear Stretching)

KURALLAR:
    - Görüntü okuma/yazma: SADECE cv2.imread / cv2.imwrite / cv2.imshow
    - Tüm işlemler NumPy vektörize operasyonları ve temel Python ile yapılmaktadır.
    - OpenCV'nin hazır algoritmaları (cvtColor, threshold, calcHist, equalizeHist vb.)
      KESİNLİKLE KULLANILMAMAKTADIR.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================
# GÖREV 4.1 — Gri Tonlama Dönüşümü
# ===========================================================================

def gri_tonlama(bgr_goruntu: np.ndarray) -> np.ndarray:
    """
    BGR formatındaki bir görüntüyü gri tonlamalı (grayscale) görüntüye dönüştürür.

    ITU-R BT.601 luminans formülü kullanılır:
        Gray = 0.299 * R + 0.587 * G + 0.114 * B

    OpenCV görüntüleri BGR sırasıyla okuyacağından kanal sırası buna göre
    ayarlanmıştır: bgr_goruntu[..., 0] = B, [1] = G, [2] = R

    Parametreler
    ------------
    bgr_goruntu : np.ndarray
        cv2.imread ile okunmuş BGR formatındaki uint8 görüntü dizisi.

    Döndürür
    --------
    np.ndarray
        uint8 tipinde, tek kanallı gri tonlamalı görüntü matrisi.
    """
    # Kanalları ayır: BGR -> B, G, R
    B = bgr_goruntu[:, :, 0].astype(np.float64)
    G = bgr_goruntu[:, :, 1].astype(np.float64)
    R = bgr_goruntu[:, :, 2].astype(np.float64)

    # Luminans ağırlıklı toplam (vektörize matris işlemi)
    gri = 0.299 * R + 0.587 * G + 0.114 * B

    # Ondalık değerleri uint8 aralığına (0-255) yuvarla ve dönüştür
    gri_uint8 = gri.astype(np.uint8)

    return gri_uint8


# ===========================================================================
# GÖREV 4.2 — Binary (İkili) Dönüşüm
# ===========================================================================

def binary_donusum(gri_goruntu: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Gri tonlamalı görüntüyü verilen eşik (threshold) değerine göre ikili
    (binary) görüntüye dönüştürür.

    İşlem:
        piksel >= threshold  →  255 (beyaz)
        piksel <  threshold  →  0   (siyah)

    Parametreler
    ------------
    gri_goruntu : np.ndarray
        Tek kanallı, uint8 gri tonlamalı görüntü matrisi.
    threshold : int, opsiyonel
        Eşik değeri (varsayılan: 127, aralık: 0-255).

    Döndürür
    --------
    np.ndarray
        uint8 tipinde ikili (0 veya 255) görüntü matrisi.
    """
    # np.where: koşul sağlanıyorsa 255, değilse 0 ata (tam vektörize)
    binary = np.where(gri_goruntu >= threshold, 255, 0).astype(np.uint8)

    return binary


# ===========================================================================
# GÖREV 4.6 — Renk Uzayı Dönüşümü: BGR -> HSV
# ===========================================================================

def bgr_to_hsv(bgr_goruntu: np.ndarray) -> np.ndarray:
    """
    BGR formatındaki görüntüyü HSV (Hue-Saturation-Value) renk uzayına dönüştürür.

    Adımlar:
        1. Piksel değerlerini [0, 1] aralığına normalize et (float64).
        2. Her piksel için Cmax, Cmin ve Delta hesapla (vektörize).
        3. Value (V) = Cmax
        4. Saturation (S):
               Delta == 0  → 0
               Delta != 0  → Delta / Cmax
        5. Hue (H) – Delta'nın hangi kanaldan geldiğine göre 6 bölgeli formül:
               Cmax == R   → H = 60 * ((G - B) / Delta mod 6)
               Cmax == G   → H = 60 * ((B - R) / Delta + 2)
               Cmax == B   → H = 60 * ((R - G) / Delta + 4)
               Delta == 0  → H = 0
        6. OpenCV uyumlu ölçekleme:
               H  : [0°, 360°) → [0, 179]   (H / 2)
               S  : [0, 1]     → [0, 255]
               V  : [0, 1]     → [0, 255]

    Parametreler
    ------------
    bgr_goruntu : np.ndarray
        cv2.imread ile okunmuş BGR formatında uint8 NumPy dizisi.

    Döndürür
    --------
    np.ndarray
        uint8 tipinde HSV formatında görüntü matrisi (H:0-179, S:0-255, V:0-255).
    """
    # --- 1. Normalize et: [0, 255] -> [0.0, 1.0] ---
    img = bgr_goruntu.astype(np.float64) / 255.0

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    # --- 2. Her piksel için maksimum ve minimum kanal değerleri ---
    Cmax = np.maximum(np.maximum(R, G), B)   # Value (parlaklık)
    Cmin = np.minimum(np.minimum(R, G), B)
    Delta = Cmax - Cmin                       # Kroma (chroma)

    # --- 3. Value ---
    V = Cmax  # shape: (H, W)

    # --- 4. Saturation ---
    # Cmax == 0 olan piksellerde bölme hatası almamak için güvenli bölme
    S = np.where(Delta == 0, 0.0, Delta / np.where(Cmax == 0, 1.0, Cmax))

    # --- 5. Hue (derece cinsinden, 0-360) ---
    # Geçici H dizisi sıfırla
    H = np.zeros_like(R)

    # Cmax == R alanı
    mask_R = (Cmax == R) & (Delta != 0)
    H[mask_R] = 60.0 * (((G[mask_R] - B[mask_R]) / Delta[mask_R]) % 6)

    # Cmax == G alanı
    mask_G = (Cmax == G) & (Delta != 0)
    H[mask_G] = 60.0 * ((B[mask_G] - R[mask_G]) / Delta[mask_G] + 2)

    # Cmax == B alanı
    mask_B = (Cmax == B) & (Delta != 0)
    H[mask_B] = 60.0 * ((R[mask_B] - G[mask_B]) / Delta[mask_B] + 4)

    # H negatif çıkabilir (mod işlemi nedeniyle nadir), 360 ekleyerek düzelt
    H = np.where(H < 0, H + 360.0, H)

    # --- 6. OpenCV uyumlu ölçekleme ---
    H_scaled = (H / 2.0).astype(np.uint8)          # [0, 360) -> [0, 179]
    S_scaled = (S * 255.0).astype(np.uint8)         # [0, 1]   -> [0, 255]
    V_scaled = (V * 255.0).astype(np.uint8)         # [0, 1]   -> [0, 255]

    # Kanalları birleştir: shape (H, W, 3)
    hsv = np.stack([H_scaled, S_scaled, V_scaled], axis=2)

    return hsv


# ===========================================================================
# GÖREV 4.7 — Histogram Hesaplama ve Histogram Germe
# ===========================================================================

def histogram_hesapla(gri_goruntu: np.ndarray) -> np.ndarray:
    """
    Gri tonlamalı görüntünün histogramını NumPy boolean sayımı ile hesaplar.
    Hazır histogram fonksiyonları (np.histogram, cv2.calcHist vb.) KULLANILMAZ.

    Her yoğunluk seviyesi i (0-255) için:
        hist[i] = (gri_goruntu == i).sum()

    Parametreler
    ------------
    gri_goruntu : np.ndarray
        Tek kanallı, uint8 gri tonlamalı görüntü matrisi.

    Döndürür
    --------
    np.ndarray
        256 elemanlı int64 histogram dizisi.
    """
    # 256 yoğunluk seviyesi için sıfır dolu dizi
    hist = np.zeros(256, dtype=np.int64)

    # Her yoğunluk değeri için boolean maske ve piksel sayısı
    for i in range(256):
        hist[i] = (gri_goruntu == i).sum()

    return hist


def histogram_germe(gri_goruntu: np.ndarray):
    """
    Doğrusal kontrast germe (Linear Histogram Stretching) uygular ve
    orijinal ile gerilmiş görüntünün histogramlarını yan yana çizer.

    Germe formülü:
        gerilmis_piksel = (piksel - min_val) / (max_val - min_val) * 255

    min_val == max_val durumunda (tek renk görüntü) bölme hatasını önlemek
    için görüntü değiştirilmeden döndürülür.

    Parametreler
    ------------
    gri_goruntu : np.ndarray
        Tek kanallı, uint8 gri tonlamalı görüntü matrisi.

    Döndürür
    --------
    gerilmis : np.ndarray
        Kontrast gerilmiş uint8 gri tonlamalı görüntü matrisi.
    """
    # --- Orijinal histogramı hesapla ---
    hist_orijinal = histogram_hesapla(gri_goruntu)

    # --- Min ve Max piksel değerlerini bul (hazır min/max kullanmak serbest) ---
    min_val = int(gri_goruntu.min())
    max_val = int(gri_goruntu.max())

    # --- Doğrusal Germe ---
    if max_val == min_val:
        # Tüm pikseller aynı: germe yapılamaz, orijinal döndürülür
        print("[UYARI] Görüntü tekdüze (min == max). Germe uygulanamıyor.")
        gerilmis = gri_goruntu.copy()
    else:
        # Piksel değerlerini float64'e çekerek hassas bölme yap
        img_float = gri_goruntu.astype(np.float64)

        # Vektörize germe formülü (her piksele aynı anda uygulanır)
        gerilmis_float = (img_float - min_val) / (max_val - min_val) * 255.0

        # Kırpma (olası yuvarlama taşmalarına karşı) ve uint8'e dönüştür
        gerilmis = np.clip(gerilmis_float, 0, 255).astype(np.uint8)

    # --- Gerilmiş görüntünün histogramını hesapla ---
    hist_gerilmis = histogram_hesapla(gerilmis)

    # --- Yan Yana Histogram Çizimi ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Histogram Germe (Linear Stretching)", fontsize=14, fontweight="bold")

    # Sol: Orijinal histogram
    axes[0].bar(range(256), hist_orijinal, color="steelblue", width=1)
    axes[0].set_title(f"Orijinal Histogram\n(min={min_val}, max={max_val})")
    axes[0].set_xlabel("Piksel Yoğunluğu (0-255)")
    axes[0].set_ylabel("Piksel Sayısı")
    axes[0].set_xlim([0, 255])

    # Sağ: Gerilmiş histogram
    axes[1].bar(range(256), hist_gerilmis, color="tomato", width=1)
    axes[1].set_title("Gerilmiş Histogram\n(min=0, max=255 hedeflenir)")
    axes[1].set_xlabel("Piksel Yoğunluğu (0-255)")
    axes[1].set_ylabel("Piksel Sayısı")
    axes[1].set_xlim([0, 255])

    plt.tight_layout()
    plt.show()

    return gerilmis


# ===========================================================================
# ANA PROGRAM — Tüm fonksiyonları sırayla test eder
# ===========================================================================

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Görüntü yükle (yolu kendi görüntünüze göre değiştirin)
    # -----------------------------------------------------------------------
    GORUNTU_YOLU = "test_goruntu.jpg"   # <-- buraya kendi görüntü yolunuzu yazın

    bgr = cv2.imread(GORUNTU_YOLU)
    if bgr is None:
        raise FileNotFoundError(
            f"Görüntü yüklenemedi: '{GORUNTU_YOLU}'\n"
            "Lütfen GORUNTU_YOLU değişkenini geçerli bir dosya yoluyla güncelleyin."
        )

    print(f"Görüntü boyutu (H x W x C): {bgr.shape}")

    # -----------------------------------------------------------------------
    # 4.1 — Gri Tonlama
    # -----------------------------------------------------------------------
    gri = gri_tonlama(bgr)
    cv2.imwrite("cikti_4_1_gri.png", gri)
    print("[4.1] Gri tonlama tamamlandı -> cikti_4_1_gri.png")

    # -----------------------------------------------------------------------
    # 4.2 — Binary Dönüşüm
    # -----------------------------------------------------------------------
    binary = binary_donusum(gri, threshold=127)
    cv2.imwrite("cikti_4_2_binary.png", binary)
    print("[4.2] Binary dönüşüm tamamlandı -> cikti_4_2_binary.png")

    # -----------------------------------------------------------------------
    # 4.6 — BGR -> HSV
    # -----------------------------------------------------------------------
    hsv = bgr_to_hsv(bgr)
    cv2.imwrite("cikti_4_6_hsv.png", hsv)
    print("[4.6] BGR->HSV dönüşümü tamamlandı -> cikti_4_6_hsv.png")

    # -----------------------------------------------------------------------
    # 4.7 — Histogram & Germe
    # -----------------------------------------------------------------------
    gerilmis = histogram_germe(gri)
    cv2.imwrite("cikti_4_7_gerilmis.png", gerilmis)
    print("[4.7] Histogram germe tamamlandı -> cikti_4_7_gerilmis.png")

    # Sonuçları ekranda göster
    cv2.imshow("Orijinal (BGR)", bgr)
    cv2.imshow("4.1 Gri Tonlama", gri)
    cv2.imshow("4.2 Binary", binary)
    cv2.imshow("4.6 HSV", hsv)
    cv2.imshow("4.7 Gerilmis", gerilmis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

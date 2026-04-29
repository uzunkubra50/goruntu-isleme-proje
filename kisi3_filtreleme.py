"""
=============================================================
KİŞİ 3 - Filtreleme & Konvolüsyon Modülü
=============================================================
Görevler:
  4.9  - Parlaklık & Kontrast Ayarı
  4.10 - Konvolüsyon & Gauss Filtresi
  4.14 - Bulanıklaştırma Filtreleri (Mean + Gauss)

Kurallar:
  ✅ Kullanılan: numpy, cv2.imread, cv2.imwrite, cv2.imshow, matplotlib
  ❌ YASAK   : cv2.GaussianBlur, cv2.blur, cv2.filter2D ve diğer hazır algoritmalar
=============================================================
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')   # GUI olmayan ortamlarda grafik kaydetmek için
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────────────────
# 4.9  PARLAKLIK & KONTRAST AYARI
# ─────────────────────────────────────────────────────────

def parlaklik_kontrast_ayari(image, alpha=1.0, beta=0):
    """
    Parlaklık ve kontrast ayarı.
      output = clip(alpha * image + beta, 0, 255)

    Parametreler:
      image : numpy array (H x W) veya (H x W x C) — uint8
      alpha : kontrast çarpanı  (örn: 1.5 → daha fazla kontrast)
      beta  : parlaklık sabiti  (örn: 50  → daha aydınlık)

    Dönüş:
      uint8 numpy array (aynı boyut)
    """
    # uint8 ile çalışırken taşma olmaması için int32'ye çevir
    img_int32 = image.astype(np.int32)
    sonuc = alpha * img_int32 + beta
    # [0, 255] aralığına sıkıştır ve uint8'e dönüştür
    sonuc = np.clip(sonuc, 0, 255).astype(np.uint8)
    return sonuc


# ─────────────────────────────────────────────────────────
# 4.10  KONVOLÜSYON & GAUSS FİLTRESİ
# ─────────────────────────────────────────────────────────

def gauss_kernel_olustur(kernel_boyutu, sigma):
    """
    2D Gauss kerneli sıfırdan oluşturur ve normalize eder.
      G(x,y) = (1 / (2π·σ²)) · exp(-(x²+y²) / (2σ²))

    Parametreler:
      kernel_boyutu : tek sayı olmalı (örn: 3, 5, 7)
      sigma         : Gauss standart sapması

    Dönüş:
      float64 numpy array (kernel_boyutu x kernel_boyutu), toplamı 1
    """
    if kernel_boyutu % 2 == 0:
        raise ValueError("Kernel boyutu tek sayı olmalıdır (3, 5, 7, ...)!")

    yari = kernel_boyutu // 2
    kernel = np.zeros((kernel_boyutu, kernel_boyutu), dtype=np.float64)

    for i in range(kernel_boyutu):
        for j in range(kernel_boyutu):
            x = j - yari  # sütun ekseni
            y = i - yari  # satır ekseni
            kernel[i, j] = (1.0 / (2 * np.pi * sigma ** 2)) * \
                            np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Normalizasyon: tüm ağırlıklar toplamı 1 olacak şekilde böl
    kernel /= kernel.sum()
    return kernel


def konvolusyon_uygula(image, kernel):
    """
    2D konvolüsyon — sıfırdan (zero-padding ile).
    Renkli (3 kanallı) görüntülerde her kanalı ayrı işler.

    Parametreler:
      image  : numpy array (H x W) veya (H x W x 3) — uint8
      kernel : 2D float numpy array

    Dönüş:
      uint8 numpy array (aynı boyut)
    """
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    def _tek_kanal_konv(kanal, krn):
        """Tek kanallı (gri) görüntü üzerinde konvolüsyon uygular."""
        h, w = kanal.shape
        # Zero-padding ekle
        padded = np.pad(kanal, ((pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant', constant_values=0)
        cikti = np.zeros((h, w), dtype=np.float64)

        for i in range(h):
            for j in range(w):
                # Komşuluk bölgesini al
                bolge = padded[i: i + k_h, j: j + k_w]
                # Kernel ile eleman bazlı çarp ve topla
                cikti[i, j] = np.sum(bolge * krn)

        return np.clip(cikti, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        # Gri görüntü
        return _tek_kanal_konv(image, kernel)
    else:
        # Renkli görüntü: her kanalı ayrı işle
        kanallar = []
        for k in range(image.shape[2]):
            kanallar.append(_tek_kanal_konv(image[:, :, k], kernel))
        return np.stack(kanallar, axis=2)


def gauss_filtresi(image, kernel_boyutu=5, sigma=1.0):
    """
    Gauss filtresi uygular (konvolüsyon + Gauss kerneli).

    Parametreler:
      image         : numpy array (H x W) veya (H x W x 3) — uint8
      kernel_boyutu : tek sayı (varsayılan 5)
      sigma         : Gauss sigma (varsayılan 1.0)

    Dönüş:
      uint8 numpy array
    """
    kernel = gauss_kernel_olustur(kernel_boyutu, sigma)
    return konvolusyon_uygula(image, kernel)


# ─────────────────────────────────────────────────────────
# 4.14  BULANIKLAŞTIRMA FİLTRELERİ
# ─────────────────────────────────────────────────────────

def mean_blur(image, kernel_boyutu=3):
    """
    Ortalama (Mean / Box) bulanıklaştırma filtresi.
    Eşit ağırlıklı kutu kerneli kullanır: her eleman = 1 / (k * k)

    Parametreler:
      image         : numpy array (H x W) veya (H x W x 3) — uint8
      kernel_boyutu : tek sayı (varsayılan 3)

    Dönüş:
      uint8 numpy array
    """
    if kernel_boyutu % 2 == 0:
        raise ValueError("Kernel boyutu tek sayı olmalıdır!")

    # Kutu (box) kerneli oluştur
    k = kernel_boyutu
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    return konvolusyon_uygula(image, kernel)


def gaussian_blur(image, kernel_boyutu=5, sigma=1.0):
    """
    Gauss bulanıklaştırma filtresi.
    Merkez piksele daha fazla, uzak piksellere daha az ağırlık verir.

    Parametreler:
      image         : numpy array (H x W) veya (H x W x 3) — uint8
      kernel_boyutu : tek sayı (varsayılan 5)
      sigma         : Gauss sigma (varsayılan 1.0)

    Dönüş:
      uint8 numpy array
    """
    return gauss_filtresi(image, kernel_boyutu, sigma)


# ─────────────────────────────────────────────────────────
# YARDIMCI: Sonuçları görselleştir ve kaydet
# ─────────────────────────────────────────────────────────

def gorsellestir_ve_kaydet(basliklar, goruntular, dosya_adi):
    """
    Birden fazla görüntüyü yan yana gösterir ve kaydeder.

    Parametreler:
      basliklar  : list of str
      goruntular : list of numpy array (uint8)
      dosya_adi  : çıktı dosya yolu (.png)
    """
    n = len(goruntular)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, baslik, img in zip(axes, basliklar, goruntular):
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(baslik, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(dosya_adi, dpi=150)
    plt.close()
    print(f"  → Kaydedildi: {dosya_adi}")


# ─────────────────────────────────────────────────────────
# BAĞIMSIZ TEST — python kisi3_filtreleme.py ile çalıştır
# ─────────────────────────────────────────────────────────

def kisi3_calistir(goruntu_yolu, cikti_klasoru="kisi3_ciktilar"):
    """
    Tüm Kişi-3 işlemlerini çalıştırır, sonuçları klasöre kaydeder.

    Parametreler:
      goruntu_yolu  : giriş görüntüsü (.jpg / .png)
      cikti_klasoru : sonuçların kaydedileceği klasör

    Dönüş:
      dict — {"parlaklik": ..., "gauss": ..., "mean_blur": ..., "gaussian_blur": ...}
    """
    os.makedirs(cikti_klasoru, exist_ok=True)

    # Görüntüyü oku
    orijinal = cv2.imread(goruntu_yolu)
    if orijinal is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {goruntu_yolu}")

    gri = np.dot(orijinal[..., ::-1].astype(np.float64),
                 [0.299, 0.587, 0.114]).astype(np.uint8)

    print("\n[Kişi 3] İşlemler başlıyor...\n")

    # ── 4.9 Parlaklık & Kontrast ──────────────────────────
    print("4.9 Parlaklık & Kontrast Ayarı...")
    prl_artir  = parlaklik_kontrast_ayari(orijinal, alpha=1.0, beta=60)
    prl_azalt  = parlaklik_kontrast_ayari(orijinal, alpha=1.0, beta=-60)
    kontrast_y = parlaklik_kontrast_ayari(orijinal, alpha=1.5, beta=0)
    kontrast_d = parlaklik_kontrast_ayari(orijinal, alpha=0.6, beta=0)

    gorsellestir_ve_kaydet(
        ["Orijinal", "Parlaklık +60", "Parlaklık -60",
         "Kontrast α=1.5", "Kontrast α=0.6"],
        [orijinal, prl_artir, prl_azalt, kontrast_y, kontrast_d],
        os.path.join(cikti_klasoru, "4_9_parlaklik_kontrast.png")
    )
    cv2.imwrite(os.path.join(cikti_klasoru, "4_9_parlaklik_artir.png"), prl_artir)
    cv2.imwrite(os.path.join(cikti_klasoru, "4_9_kontrast_artir.png"), kontrast_y)

    # ── 4.10 Konvolüsyon & Gauss Filtresi ─────────────────
    print("4.10 Konvolüsyon & Gauss Filtresi...")
    gauss_3x3 = gauss_filtresi(gri, kernel_boyutu=3, sigma=0.8)
    gauss_5x5 = gauss_filtresi(gri, kernel_boyutu=5, sigma=1.0)
    gauss_7x7 = gauss_filtresi(gri, kernel_boyutu=7, sigma=2.0)

    gorsellestir_ve_kaydet(
        ["Gri Orijinal", "Gauss 3x3 σ=0.8",
         "Gauss 5x5 σ=1.0", "Gauss 7x7 σ=2.0"],
        [gri, gauss_3x3, gauss_5x5, gauss_7x7],
        os.path.join(cikti_klasoru, "4_10_gauss_filtre.png")
    )
    cv2.imwrite(os.path.join(cikti_klasoru, "4_10_gauss_5x5.png"), gauss_5x5)

    # Kernel görselleştir (sadece bilgi amaçlı)
    k = gauss_kernel_olustur(5, 1.0)
    print(f"  Gauss 5x5 kernel (σ=1.0):\n{np.round(k, 4)}")

    # ── 4.14 Bulanıklaştırma Filtreleri ───────────────────
    print("4.14 Bulanıklaştırma Filtreleri...")
    mean_3  = mean_blur(gri, kernel_boyutu=3)
    mean_7  = mean_blur(gri, kernel_boyutu=7)
    gauss_b = gaussian_blur(gri, kernel_boyutu=5, sigma=1.0)

    gorsellestir_ve_kaydet(
        ["Gri Orijinal", "Mean Blur 3x3",
         "Mean Blur 7x7", "Gaussian Blur 5x5"],
        [gri, mean_3, mean_7, gauss_b],
        os.path.join(cikti_klasoru, "4_14_bulaniklik_filtreleri.png")
    )
    cv2.imwrite(os.path.join(cikti_klasoru, "4_14_mean_blur.png"),  mean_3)
    cv2.imwrite(os.path.join(cikti_klasoru, "4_14_gauss_blur.png"), gauss_b)

    print("\n[Kişi 3] Tüm işlemler tamamlandı!")
    print(f"Çıktılar: {os.path.abspath(cikti_klasoru)}/\n")

    return {
        "parlaklik_artir":  prl_artir,
        "parlaklik_azalt":  prl_azalt,
        "kontrast_artir":   kontrast_y,
        "gauss_3x3":        gauss_3x3,
        "gauss_5x5":        gauss_5x5,
        "mean_blur_3x3":    mean_3,
        "gaussian_blur_5x5": gauss_b,
    }


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Kullanım: python kisi3_filtreleme.py <goruntu_yolu>")
        print("Örnek   : python kisi3_filtreleme.py test.jpg")
        sys.exit(1)

    goruntu_yolu = sys.argv[1]
    kisi3_calistir(goruntu_yolu)
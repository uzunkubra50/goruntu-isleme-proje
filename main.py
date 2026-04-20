# -*- coding: utf-8 -*-
"""
main.py
--------
Goruntu Isleme Dersi - Ana Proje Dosyasi

Aciklama:
    Bu dosya, kisi5_morfoloji.py modulundeki morfolojik islem
    fonksiyonlarini kullanarak test goruntusu uzerinde Dilation,
    Erosion, Opening ve Closing islemlerini uygular.

    Izin verilen OpenCV fonksiyonlari:
        [OK] cv2.imread   - goruntu okuma
        [OK] cv2.imwrite  - goruntu kaydetme
        [OK] cv2.imshow   - goruntu gosterme (istege bagli)

    Yasak fonksiyonlar:
        [XX] cv2.erode
        [XX] cv2.dilate
        [XX] cv2.morphologyEx
        (ve diger tum hazir morfoloji fonksiyonlari)

Yazar: Kisi 5
Tarih: 2026
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')   # GUI olmayan ortamlar icin (gerekirse degistir)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Kendi yazdigimiz morfoloji modulunu ice aktar
from kisi5_morfoloji import dilation, erosion, opening, closing


# ===========================================================================
# SABITLER VE YAPILANDIRMA
# ===========================================================================
GORUNTU_YOLU   = 'images/test.jpg'   # Giris goruntusu yolu
CIKTI_KLASORU  = 'outputs'           # Ciktilarin kaydedilecegi klasor
KERNEL_BOYUTU  = 5                   # Yapisal elemanin boyutu (5x5 kare kernel)
ESIK_DEGERI    = 127                 # Binary esikleme icin esik degeri (0-255)


# ===========================================================================
# YARDIMCI FONKSIYON: Binary (Ikili) goruntuye donusturme
# ===========================================================================
def gri_to_binary(gri_goruntu: np.ndarray, esik: int = 127) -> np.ndarray:
    """
    Gri tonlamali goruntuyu esikleme ile binary goruntuye cevirir.

    Parametreler:
        gri_goruntu (np.ndarray): Gri tonlamali giris goruntusu (tek kanal)
        esik (int)              : Piksel degeri esigi (bu degerin ustundekiler 255 olur)

    Dondurur:
        np.ndarray: 0 veya 255 degerli binary goruntu (uint8)
    """
    # esik degerinin uzerindeki pikseller 255 (beyaz), altindakiler 0 (siyah) olur
    binary = np.where(gri_goruntu > esik, 255, 0).astype(np.uint8)
    return binary


# ===========================================================================
# YARDIMCI FONKSIYON: Cikti klasorunu olusturma
# ===========================================================================
def cikti_klasoru_olustur(klasor_yolu: str) -> None:
    """
    Cikti klasorunu olusturur (zaten varsa atlar).

    Parametreler:
        klasor_yolu (str): Olusturulacak klasorun yolu
    """
    if not os.path.exists(klasor_yolu):
        os.makedirs(klasor_yolu)
        print("[OK] Klasor olusturuldu: '{}'".format(klasor_yolu))
    else:
        print("[ii] Klasor zaten mevcut: '{}'".format(klasor_yolu))


# ===========================================================================
# YARDIMCI FONKSIYON: Goruntu kaydetme
# ===========================================================================
def goruntu_kaydet(goruntu: np.ndarray, dosya_adi: str) -> None:
    """
    Goruntuyu outputs klasorune kaydeder (cv2.imwrite kullanir).

    Parametreler:
        goruntu   (np.ndarray): Kaydedilecek goruntu
        dosya_adi (str)       : Cikti dosya adi
    """
    tam_yol = os.path.join(CIKTI_KLASORU, dosya_adi)
    cv2.imwrite(tam_yol, goruntu)
    print("[OK] Goruntu kaydedildi: '{}'".format(tam_yol))


# ===========================================================================
# YARDIMCI FONKSIYON: Matplotlib ile gorsellestirme
# ===========================================================================
def sonuclari_goster(goruntular: dict, baslik: str = "Morfolojik Islem Sonuclari") -> None:
    """
    Orijinal ve islenmis goruntuleri yan yana Matplotlib ile gosterir.

    Parametreler:
        goruntular (dict): {'Goruntu Adi': numpy_dizisi} formatinda sozluk
        baslik (str)     : Pencerenin basligi
    """
    adet = len(goruntular)

    # Figur ve alt grafik duzenini olustur
    fig = plt.figure(figsize=(4 * adet, 5))
    fig.suptitle(baslik, fontsize=14, fontweight='bold', y=1.02)

    # GridSpec ile alt grafikleri duzenle
    gs = gridspec.GridSpec(1, adet, figure=fig, wspace=0.4)

    for idx, (isim, goruntu) in enumerate(goruntular.items()):
        eksen = fig.add_subplot(gs[0, idx])

        # Goruntuyu gri tonlamali olarak goster
        eksen.imshow(goruntu, cmap='gray', vmin=0, vmax=255)
        eksen.set_title(isim, fontsize=9, pad=8, fontweight='bold')
        eksen.axis('off')  # Eksen cizgilerini gizle

    # Yerlesimi duzenle ve kaydet
    plt.tight_layout()
    kayit_yolu = os.path.join(CIKTI_KLASORU, 'karsilastirma_tablosu.png')
    plt.savefig(kayit_yolu, dpi=150, bbox_inches='tight')
    print("[OK] Karsilastirma tablosu kaydedildi: '{}'".format(kayit_yolu))

    # Ekranda gostermek istersen asagidaki satiri ac:
    # plt.show()
    plt.close()


# ===========================================================================
# ANA FONKSIYON: main()
# ===========================================================================
def main():
    """
    Projenin ana akisini yonetir:
        1. Goruntuyu oku ve binary'e cevir
        2. 4 morfolojik islemi uygula
        3. Ciktilari kaydet
        4. Sonuclari gorsellestir
    """
    print("=" * 60)
    print("   MORFOLOJIK ISLEMLER - GORUNTU ISLEME PROJESI")
    print("=" * 60)

    # ------------------------------------------------------------------
    # ADIM 1: Cikti klasorunu olustur
    # ------------------------------------------------------------------
    cikti_klasoru_olustur(CIKTI_KLASORU)
    print()

    # ------------------------------------------------------------------
    # ADIM 2: Goruntuyu cv2.imread ile oku
    # ------------------------------------------------------------------
    print("[>>] Goruntu okunuyor: '{}' ...".format(GORUNTU_YOLU))

    # cv2.imread ile goruntuyu oku (BGR formatinda doner)
    goruntu_bgr = cv2.imread(GORUNTU_YOLU)

    # Dosya yolu hataliysa veya dosya yoksa hata ver ve cik
    if goruntu_bgr is None:
        print("\n[HATA] Goruntu dosyasi bulunamadi: '{}'".format(GORUNTU_YOLU))
        print("       Lutfen 'images/test.jpg' dosyasinin mevcut oldugunu kontrol edin.")
        print("       Once 'python test_goruntu_olustur.py' calistirin.")
        return

    print("[OK] Goruntu basariyla okundu. Boyut: {}".format(goruntu_bgr.shape))

    # ------------------------------------------------------------------
    # ADIM 3: BGR goruntuyu Gri tona cevir (NumPy ile, cv2.cvtColor kullanmadan)
    # Gri = 0.2989*R + 0.5870*G + 0.1140*B
    # OpenCV BGR sirasinda: B=ch[0], G=ch[1], R=ch[2]
    # ------------------------------------------------------------------
    gri_goruntu = (
        0.2989 * goruntu_bgr[:, :, 2] +   # Kirmizi (R) kanali
        0.5870 * goruntu_bgr[:, :, 1] +   # Yesil (G) kanali
        0.1140 * goruntu_bgr[:, :, 0]     # Mavi (B) kanali
    ).astype(np.uint8)

    print("[OK] Gri tona cevrildi. Boyut: {}".format(gri_goruntu.shape))

    # ------------------------------------------------------------------
    # ADIM 4: Gri goruntuyu Binary (Ikili) goruntuye cevir
    # ------------------------------------------------------------------
    binary_goruntu = gri_to_binary(gri_goruntu, esik=ESIK_DEGERI)
    print("[OK] Binary goruntu olusturuldu (esik: {})".format(ESIK_DEGERI))
    print()

    # ------------------------------------------------------------------
    # ADIM 5: Orijinal goruntuleri kaydet
    # ------------------------------------------------------------------
    goruntu_kaydet(gri_goruntu,    '00_gri_goruntu.jpg')
    goruntu_kaydet(binary_goruntu, '01_binary_goruntu.jpg')
    print()

    # ------------------------------------------------------------------
    # ADIM 6: Morfolojik islemleri uygula
    # ------------------------------------------------------------------
    print("[>>] Morfolojik islemler uygulanıyor (Kernel: {}x{}) ...".format(
        KERNEL_BOYUTU, KERNEL_BOYUTU))
    print("     Bu islem goruntu boyutuna gore biraz zaman alabilir...\n")

    # ---- 6a. GENISLEME (Dilation) ----
    print("[>>] Genisleme (Dilation) hesaplaniyor...")
    dilation_sonucu = dilation(binary_goruntu, kernel_size=KERNEL_BOYUTU)
    goruntu_kaydet(dilation_sonucu, '02_dilation_genisleme.jpg')
    print()

    # ---- 6b. ASINMA (Erosion) ----
    print("[>>] Asinma (Erosion) hesaplaniyor...")
    erosion_sonucu = erosion(binary_goruntu, kernel_size=KERNEL_BOYUTU)
    goruntu_kaydet(erosion_sonucu, '03_erosion_asinma.jpg')
    print()

    # ---- 6c. ACMA (Opening) ----
    print("[>>] Acma (Opening) hesaplaniyor...")
    opening_sonucu = opening(binary_goruntu, kernel_size=KERNEL_BOYUTU)
    goruntu_kaydet(opening_sonucu, '04_opening_acma.jpg')
    print()

    # ---- 6d. KAPAMA (Closing) ----
    print("[>>] Kapama (Closing) hesaplaniyor...")
    closing_sonucu = closing(binary_goruntu, kernel_size=KERNEL_BOYUTU)
    goruntu_kaydet(closing_sonucu, '05_closing_kapama.jpg')
    print()

    # ------------------------------------------------------------------
    # ADIM 7: Tum sonuclari Matplotlib ile gorsellestir
    # ------------------------------------------------------------------
    print("[>>] Sonuclar gorsellestiriliyor...")

    # Gosterilecek goruntulerin sozlugu (sirasiyla)
    gosterilecek_goruntular = {
        "Orijinal\n(Gri Ton)": gri_goruntu,
        "Binary\n(Ikili)": binary_goruntu,
        "Dilation\n(Genisleme)": dilation_sonucu,
        "Erosion\n(Asinma)": erosion_sonucu,
        "Opening\n(Acma)": opening_sonucu,
        "Closing\n(Kapama)": closing_sonucu,
    }

    # Matplotlib ile yan yana goster ve kaydet
    sonuclari_goster(
        goruntular=gosterilecek_goruntular,
        baslik="Morfolojik Islemler | Kernel: {}x{} | Esik: {}".format(
            KERNEL_BOYUTU, KERNEL_BOYUTU, ESIK_DEGERI)
    )

    # ------------------------------------------------------------------
    # ADIM 8: Ozet bilgisi yazdir
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("   ISLEM TAMAMLANDI!")
    print("=" * 60)
    print("Cikti dosyalari '{}/' klasorune kaydedildi:".format(CIKTI_KLASORU))
    print("  - 00_gri_goruntu.jpg          : Gri tonlamali goruntu")
    print("  - 01_binary_goruntu.jpg       : Binary (esiklenimis) goruntu")
    print("  - 02_dilation_genisleme.jpg   : Genisleme islemi sonucu")
    print("  - 03_erosion_asinma.jpg       : Asinma islemi sonucu")
    print("  - 04_opening_acma.jpg         : Acma islemi sonucu")
    print("  - 05_closing_kapama.jpg       : Kapama islemi sonucu")
    print("  - karsilastirma_tablosu.png   : Tum sonuclarin karsilastirmasi")
    print("=" * 60)


# ===========================================================================
# PROGRAM GIRIS NOKTASI
# ===========================================================================
if __name__ == "__main__":
    main()

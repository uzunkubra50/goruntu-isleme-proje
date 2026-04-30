# -*- coding: utf-8 -*-
"""
kisi2_geometrik.py — Goruntu Isleme Dersi, Grup 12
----------------------------------------------------
Kisi 2: Geometrik Islemler
  - 4.3  Goruntu Dondurme   (inverse mapping, 2D donusturme matrisi)
  - 4.4  Goruntu Kirpma     (NumPy array slicing, sinir kontrolu)
  - 4.5  Yakinlastirma/Uzaklastirma (Nearest Neighbor interpolasyon)
  - 4.8  Aritmetik Islemler (toplama, carpma — int32 donusum + clip)

KURAL: cv2.resize, cv2.rotate, cv2.warpAffine KULLANILMAZ.
       Tum islemler NumPy dongusu / matematiksel formul ile sifirdan.
"""

import numpy as np
import math


# ============================================================
# 4.3  GORUNTU DONDURME
# ============================================================

def goruntu_dondur(goruntu: np.ndarray, aci_derece: float,
                   arka_plan: int = 0) -> np.ndarray:
    """
    Goruntu dondurme — saat yonunde pozitif aci.

    Yontem: Inverse Mapping
      Cikti piksel (x', y') icin kaynak koordinat:
        x = cos(t)*(x'-cx') + sin(t)*(y'-cy') + cx
        y = -sin(t)*(x'-cx') + cos(t)*(y'-cy') + cy
      Kaynak sinir icindeyse pikseli kopyala, disindaysa arka_plan.

    Parametreler
    ------------
    goruntu     : H x W x C  veya  H x W  NumPy dizisi (uint8)
    aci_derece  : Dondurme acisi (derece). Pozitif = saat yonu.
    arka_plan   : Sinir disi pikseller icin deger (varsayilan: 0 / siyah)

    Donus
    -----
    Ayni boyutta donmus goruntu (uint8)
    """
    renkli = (goruntu.ndim == 3)
    H, W = goruntu.shape[:2]

    # Radyana cevir
    t = math.radians(aci_derece)
    cos_t = math.cos(t)
    sin_t = math.sin(t)

    # Merkez koordinatlari
    cx = W / 2.0
    cy = H / 2.0

    # Cikti boyutu orijinal ile ayni (kesilme olabilir ama proje bunu istiyor)
    if renkli:
        cikti = np.full((H, W, goruntu.shape[2]), arka_plan, dtype=np.uint8)
    else:
        cikti = np.full((H, W), arka_plan, dtype=np.uint8)

    # Cikti piksel koordinat gridleri (vektorize)
    out_y_idx, out_x_idx = np.mgrid[0:H, 0:W]   # her biri H x W

    # Merkeze gore ofset
    dx = out_x_idx - cx
    dy = out_y_idx - cy

    # Inverse rotation (geriye coz)
    src_x = (cos_t * dx + sin_t * dy + cx).astype(np.float64)
    src_y = (-sin_t * dx + cos_t * dy + cy).astype(np.float64)

    # Nearest Neighbor: tam sayiya yuvarlama
    src_xi = np.round(src_x).astype(np.int32)
    src_yi = np.round(src_y).astype(np.int32)

    # Gecerli koordinat maskesi
    gecerli = (
        (src_xi >= 0) & (src_xi < W) &
        (src_yi >= 0) & (src_yi < H)
    )

    # Pikselleri kopyala
    if renkli:
        cikti[gecerli] = goruntu[src_yi[gecerli], src_xi[gecerli]]
    else:
        cikti[gecerli] = goruntu[src_yi[gecerli], src_xi[gecerli]]

    return cikti


# ============================================================
# 4.4  GORUNTU KIRPMA
# ============================================================

def goruntu_kirp(goruntu: np.ndarray,
                 x1: int, y1: int,
                 x2: int, y2: int) -> np.ndarray:
    """
    Goruntu kirpma — NumPy dizi dilimleme ile.

    Koordinatlar otomatik olarak goruntu sinirlarina kisilir;
    gecersiz aralik (x1 >= x2 vb.) durumunda orijinal goruntu doner.

    Parametreler
    ------------
    goruntu : NumPy uint8 dizisi
    x1, y1  : Sol-ust kosesi (dahil)
    x2, y2  : Sag-alt kosesi (dahil degil)

    Donus
    -----
    Kirpilmis goruntu (uint8)
    """
    H, W = goruntu.shape[:2]

    # Sinir kontrolu
    x1 = int(max(0, min(x1, W - 1)))
    y1 = int(max(0, min(y1, H - 1)))
    x2 = int(max(0, min(x2, W)))
    y2 = int(max(0, min(y2, H)))

    if x1 >= x2 or y1 >= y2:
        # Gecersiz aralik: orijinali dondur
        return goruntu.copy()

    # Temel NumPy dilimleme — hazir fonksiyon yok
    kirpilmis = goruntu[y1:y2, x1:x2]
    return kirpilmis.copy()


# ============================================================
# 4.5  YAKINLASTIRMA / UZAKLASTIRMA  (Nearest Neighbor)
# ============================================================

def goruntu_olcekle(goruntu: np.ndarray,
                    olcek_x: float,
                    olcek_y: float = None) -> np.ndarray:
    """
    En Yakin Komsu (Nearest Neighbor) interpolasyon ile olcekleme.

    Formul:
        src_x = int(out_x * (src_W / out_W))
        src_y = int(out_y * (src_H / out_H))

    Parametreler
    ------------
    goruntu : NumPy uint8 dizisi (renkli veya gri)
    olcek_x : Yatay olcek faktoru (ornek: 2.0 = 2 kat buyut, 0.5 = yariya indir)
    olcek_y : Dikey olcek faktoru (None ise olcek_x kullanilir)

    Donus
    -----
    Olceklenmis goruntu (uint8)
    """
    if olcek_y is None:
        olcek_y = olcek_x

    renkli = (goruntu.ndim == 3)
    src_H, src_W = goruntu.shape[:2]

    # Cikti boyutu
    out_W = max(1, int(round(src_W * olcek_x)))
    out_H = max(1, int(round(src_H * olcek_y)))

    # Cikti koordinat gridleri
    out_y_idx, out_x_idx = np.mgrid[0:out_H, 0:out_W]

    # Kaynak koordinatlar (Nearest Neighbor)
    src_xi = (out_x_idx * (src_W / out_W)).astype(np.int32)
    src_yi = (out_y_idx * (src_H / out_H)).astype(np.int32)

    # Guvenli sinir kirp
    src_xi = np.clip(src_xi, 0, src_W - 1)
    src_yi = np.clip(src_yi, 0, src_H - 1)

    # Pikselleri kopyala
    if renkli:
        cikti = goruntu[src_yi, src_xi]          # H x W x C
    else:
        cikti = goruntu[src_yi, src_xi]          # H x W

    return cikti.astype(np.uint8)


# ============================================================
# 4.8  ARİTMETİK İŞLEMLER
# ============================================================

def goruntu_topla(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    İki goruntu toplanimi — taşma kontrollu.

    Formul: result = clip(int32(img1) + int32(img2), 0, 255)

    Goruntulerin boyutlari esit olmayabilir;
    kucuk olan buyuge sifir-doldurularak (zero-pad) esitlenir.

    Parametreler
    ------------
    img1, img2 : uint8 NumPy dizileri (ayni kanalda olmali)

    Donus
    -----
    Toplamli goruntu (uint8)
    """
    img1, img2 = _boyut_esitle(img1, img2)

    toplam = img1.astype(np.int32) + img2.astype(np.int32)
    return np.clip(toplam, 0, 255).astype(np.uint8)


def goruntu_carp(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    İki goruntu carpimi — normalize edilmis.

    Formul: result = clip((float64(img1) * float64(img2)) / 255, 0, 255)

    Bolme (/ 255) sonucu uint8 aralığına (0-255) çekmek içindir.

    Parametreler
    ------------
    img1, img2 : uint8 NumPy dizileri

    Donus
    -----
    Carpimli goruntu (uint8)
    """
    img1, img2 = _boyut_esitle(img1, img2)

    carpim = (img1.astype(np.float64) * img2.astype(np.float64)) / 255.0
    return np.clip(carpim, 0, 255).astype(np.uint8)


def goruntu_fark(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Mutlak fark goruntüsü — iki goruntu arasindaki farki gosterir.

    Formul: result = clip(|int32(img1) - int32(img2)|, 0, 255)
    """
    img1, img2 = _boyut_esitle(img1, img2)

    fark = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
    return np.clip(fark, 0, 255).astype(np.uint8)


# ---- Yardimci: iki goruntunun boyutunu esitler ----
def _boyut_esitle(img1: np.ndarray, img2: np.ndarray):
    """
    İki goruntunun boyutunu (H, W) bakimindan esitler.
    Kucuk goruntu buyuk goruntunun sol-ust kosesine sifir-doldurulur.
    Kanal sayisi farkli ise ikisi de griye cekilir.
    """
    # Kanal uyumu
    if img1.ndim != img2.ndim:
        if img1.ndim == 3:
            img1 = np.mean(img1, axis=2).astype(np.uint8)
        if img2.ndim == 3:
            img2 = np.mean(img2, axis=2).astype(np.uint8)

    H = max(img1.shape[0], img2.shape[0])
    W = max(img1.shape[1], img2.shape[1])

    def _pad(img, H, W):
        if img.ndim == 3:
            padded = np.zeros((H, W, img.shape[2]), dtype=np.uint8)
        else:
            padded = np.zeros((H, W), dtype=np.uint8)
        padded[:img.shape[0], :img.shape[1]] = img
        return padded

    img1 = _pad(img1, H, W)
    img2 = _pad(img2, H, W)
    return img1, img2


# ============================================================
# BASIT TEST  (python kisi2_geometrik.py ile calistirin)
# ============================================================

if __name__ == "__main__":
    import cv2

    # Test goruntüsü olustur
    test = np.zeros((200, 300, 3), dtype=np.uint8)
    test[50:150, 80:220] = [0, 128, 255]   # Turuncu dikdortgen
    test[70:130, 120:180] = [255, 0, 0]    # Mavi kare

    print("=== KISI 2 — Geometrik Islemler Testi ===\n")

    # 1. Dondurme
    dondurulmus = goruntu_dondur(test, aci_derece=45)
    cv2.imwrite("cikti_dondurme_45.jpg", dondurulmus)
    print("Dondurme (45 derece) kaydedildi: cikti_dondurme_45.jpg")

    dondurulmus_90 = goruntu_dondur(test, aci_derece=90)
    cv2.imwrite("cikti_dondurme_90.jpg", dondurulmus_90)
    print("Dondurme (90 derece) kaydedildi: cikti_dondurme_90.jpg")

    # 2. Kirpma
    kirpilmis = goruntu_kirp(test, x1=80, y1=50, x2=220, y2=150)
    cv2.imwrite("cikti_kirpma.jpg", kirpilmis)
    print(f"Kirpma kaydedildi: cikti_kirpma.jpg  -> boyut {kirpilmis.shape}")

    # 3. Olcekleme
    buyutulmus = goruntu_olcekle(test, olcek_x=2.0)
    cv2.imwrite("cikti_buyut.jpg", buyutulmus)
    print(f"Buyutme (2x) kaydedildi: cikti_buyut.jpg  -> boyut {buyutulmus.shape}")

    kucultulmus = goruntu_olcekle(test, olcek_x=0.5)
    cv2.imwrite("cikti_kucult.jpg", kucultulmus)
    print(f"Kucultme (0.5x) kaydedildi: cikti_kucult.jpg  -> boyut {kucultulmus.shape}")

    # 4. Aritmetik
    katman = np.zeros_like(test)
    katman[:, :, 0] = 80   # Mavi kanal uzerine +80
    toplam  = goruntu_topla(test, katman)
    fark    = goruntu_fark(test, katman)
    carpim  = goruntu_carp(test, (test * 0.7).astype(np.uint8))

    cv2.imwrite("cikti_toplama.jpg", toplam)
    cv2.imwrite("cikti_fark.jpg",    fark)
    cv2.imwrite("cikti_carpma.jpg",  carpim)
    print("Aritmetik islemler kaydedildi.")

    print("\nTum testler basariyla tamamlandi!")
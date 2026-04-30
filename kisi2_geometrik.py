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
    Inverse Mapping yontemi ile.
    """
    renkli = (goruntu.ndim == 3)
    H, W = goruntu.shape[:2]

    t = math.radians(aci_derece)
    cos_t = math.cos(t)
    sin_t = math.sin(t)

    cx = W / 2.0
    cy = H / 2.0

    if renkli:
        cikti = np.full((H, W, goruntu.shape[2]), arka_plan, dtype=np.uint8)
    else:
        cikti = np.full((H, W), arka_plan, dtype=np.uint8)

    out_y_idx, out_x_idx = np.mgrid[0:H, 0:W]

    dx = out_x_idx - cx
    dy = out_y_idx - cy

    src_x = cos_t * dx + sin_t * dy + cx
    src_y = -sin_t * dx + cos_t * dy + cy

    src_xi = np.round(src_x).astype(np.int32)
    src_yi = np.round(src_y).astype(np.int32)

    gecerli = (
        (src_xi >= 0) & (src_xi < W) &
        (src_yi >= 0) & (src_yi < H)
    )

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
    Koordinatlar otomatik olarak goruntu sinirlarina kisilir.
    """
    H, W = goruntu.shape[:2]

    x1 = int(max(0, min(x1, W - 1)))
    y1 = int(max(0, min(y1, H - 1)))
    x2 = int(max(0, min(x2, W)))
    y2 = int(max(0, min(y2, H)))

    if x1 >= x2 or y1 >= y2:
        return goruntu.copy()

    return goruntu[y1:y2, x1:x2].copy()


# ============================================================
# 4.5  YAKINLASTIRMA / UZAKLASTIRMA  (Nearest Neighbor)
# ============================================================

def goruntu_olcekle(goruntu: np.ndarray,
                    olcek_x: float,
                    olcek_y: float = None) -> np.ndarray:
    """
    En Yakin Komsu (Nearest Neighbor) interpolasyon ile olcekleme.

    Formul:
        out_W = int(src_W * olcek_x)
        out_H = int(src_H * olcek_y)
        src_x = int(out_x / olcek_x)
        src_y = int(out_y / olcek_y)
    """
    if olcek_y is None:
        olcek_y = olcek_x

    renkli = (goruntu.ndim == 3)
    src_H, src_W = goruntu.shape[:2]

    out_W = max(1, int(src_W * olcek_x))
    out_H = max(1, int(src_H * olcek_y))

    out_y_idx, out_x_idx = np.mgrid[0:out_H, 0:out_W]

    src_xi = (out_x_idx / olcek_x).astype(np.int32)
    src_yi = (out_y_idx / olcek_y).astype(np.int32)

    src_xi = np.clip(src_xi, 0, src_W - 1)
    src_yi = np.clip(src_yi, 0, src_H - 1)

    if renkli:
        cikti = goruntu[src_yi, src_xi]
    else:
        cikti = goruntu[src_yi, src_xi]

    return cikti.astype(np.uint8)


def goruntu_yakinlastir(goruntu: np.ndarray, olcek: float) -> np.ndarray:
    """
    Yakinlastirma: Goruntu buyutulur, merkezi kirpilarak
    orijinal boyuta getirilir. Kutuda zoom etkisi gorulur.
    """
    buyuk = goruntu_olcekle(goruntu, olcek_x=olcek)
    H_b, W_b = buyuk.shape[:2]
    H_o, W_o = goruntu.shape[:2]

    y1 = max(0, (H_b - H_o) // 2)
    x1 = max(0, (W_b - W_o) // 2)
    y2 = min(y1 + H_o, H_b)
    x2 = min(x1 + W_o, W_b)

    return buyuk[y1:y2, x1:x2].copy()


def goruntu_uzaklastir(goruntu: np.ndarray, olcek: float) -> np.ndarray:
    """
    Uzaklastirma: Goruntu kucultulur, orijinal boyutta
    siyah zemine ortaya yerlestirilir. Uzaklasma etkisi gorulur.
    """
    kucuk = goruntu_olcekle(goruntu, olcek_x=olcek)
    H_k, W_k = kucuk.shape[:2]
    H_o, W_o = goruntu.shape[:2]

    if goruntu.ndim == 3:
        zemin = np.zeros((H_o, W_o, goruntu.shape[2]), dtype=np.uint8)
    else:
        zemin = np.zeros((H_o, W_o), dtype=np.uint8)

    y_bas = (H_o - H_k) // 2
    x_bas = (W_o - W_k) // 2

    zemin[y_bas:y_bas+H_k, x_bas:x_bas+W_k] = kucuk

    return zemin


# ============================================================
# 4.8  ARİTMETİK İŞLEMLER
# ============================================================

def goruntu_topla(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Toplama: result = clip(int32(img1) + int32(img2), 0, 255)"""
    img1, img2 = _boyut_esitle(img1, img2)
    return np.clip(img1.astype(np.int32) + img2.astype(np.int32), 0, 255).astype(np.uint8)


def goruntu_carp(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Carpma: result = clip((float64(img1) * float64(img2)) / 255, 0, 255)"""
    img1, img2 = _boyut_esitle(img1, img2)
    return np.clip((img1.astype(np.float64) * img2.astype(np.float64)) / 255.0, 0, 255).astype(np.uint8)


def goruntu_fark(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Mutlak fark: result = clip(|int32(img1) - int32(img2)|, 0, 255)"""
    img1, img2 = _boyut_esitle(img1, img2)
    return np.clip(np.abs(img1.astype(np.int32) - img2.astype(np.int32)), 0, 255).astype(np.uint8)


def _boyut_esitle(img1: np.ndarray, img2: np.ndarray):
    if img1.ndim != img2.ndim:
        if img1.ndim == 3:
            img1 = np.mean(img1, axis=2).astype(np.uint8)
        if img2.ndim == 3:
            img2 = np.mean(img2, axis=2).astype(np.uint8)
    H = max(img1.shape[0], img2.shape[0])
    W = max(img1.shape[1], img2.shape[1])

    def _pad(img, H, W):
        if img.ndim == 3:
            p = np.zeros((H, W, img.shape[2]), dtype=np.uint8)
        else:
            p = np.zeros((H, W), dtype=np.uint8)
        p[:img.shape[0], :img.shape[1]] = img
        return p

    return _pad(img1, H, W), _pad(img2, H, W)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    import cv2

    test = np.zeros((200, 300, 3), dtype=np.uint8)
    test[50:150, 80:220] = [0, 128, 255]
    test[70:130, 120:180] = [255, 0, 0]

    print("=== KISI 2 — Test ===\n")

    cv2.imwrite("cikti_dondurme_45.jpg",  goruntu_dondur(test, 45))
    cv2.imwrite("cikti_dondurme_90.jpg",  goruntu_dondur(test, 90))
    cv2.imwrite("cikti_kirpma.jpg",       goruntu_kirp(test, 80, 50, 220, 150))
    cv2.imwrite("cikti_yakinlastir.jpg",  goruntu_yakinlastir(test, 2.0))
    cv2.imwrite("cikti_uzaklastir.jpg",   goruntu_uzaklastir(test, 0.5))

    katman = np.full_like(test, 60)
    cv2.imwrite("cikti_toplama.jpg", goruntu_topla(test, katman))
    cv2.imwrite("cikti_fark.jpg",    goruntu_fark(test, katman))
    cv2.imwrite("cikti_carpma.jpg",  goruntu_carp(test, (test * 0.7).astype(np.uint8)))

    print("Tum dosyalar olusturuldu!")
    for isim in ["cikti_yakinlastir.jpg", "cikti_uzaklastir.jpg"]:
        img = cv2.imread(isim)
        if img is not None:
            print(f"{isim} -> {img.shape[1]}x{img.shape[0]} piksel")

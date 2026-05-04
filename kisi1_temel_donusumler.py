# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 4.1 - Gri Tonlama Dönüşümü
def gri_tonlama(bgr_goruntu):
    # Formül: 0.299*R + 0.587*G + 0.114*B
    B = bgr_goruntu[:, :, 0].astype(np.float64)
    G = bgr_goruntu[:, :, 1].astype(np.float64)
    R = bgr_goruntu[:, :, 2].astype(np.float64)

    gri = 0.299 * R + 0.587 * G + 0.114 * B
    return gri.astype(np.uint8)

# 4.2 - Binary (İkili) Dönüşüm
def binary_donusum(gri_goruntu, threshold=127):
    # Eşik değerinin üstündekileri beyaz (255), altındakileri siyah (0) yapıyoruz
    binary = np.where(gri_goruntu >= threshold, 255, 0).astype(np.uint8)
    return binary

# 4.6 - BGR -> HSV Dönüşümü
def bgr_to_hsv(bgr_goruntu):
    # Değerleri 0-1 arasına çekiyoruz
    img = bgr_goruntu.astype(np.float64) / 255.0
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]

    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    Delta = Cmax - Cmin

    # V (Value) değeri en büyük kanal değeridir
    V = Cmax
    # S (Saturation) değeri Delta/Cmax formülüyle bulunur
    S = np.where(Delta == 0, 0.0, Delta / np.where(Cmax == 0, 1.0, Cmax))

    # H (Hue) hesaplaması Delta'nın hangi kanaldan geldiğine göre değişir
    H = np.zeros_like(R)
    mask_R = (Cmax == R) & (Delta != 0)
    H[mask_R] = 60.0 * (((G[mask_R] - B[mask_R]) / Delta[mask_R]) % 6)
    mask_G = (Cmax == G) & (Delta != 0)
    H[mask_G] = 60.0 * ((B[mask_G] - R[mask_G]) / Delta[mask_G] + 2)
    mask_B = (Cmax == B) & (Delta != 0)
    H[mask_B] = 60.0 * ((R[mask_B] - G[mask_B]) / Delta[mask_B] + 4)
    
    H = np.where(H < 0, H + 360.0, H)

    # OpenCV standartlarına uygun ölçekleme (H/2, S*255, V*255)
    return np.stack([(H/2).astype(np.uint8), (S*255).astype(np.uint8), (V*255).astype(np.uint8)], axis=2)

# 4.7 - Histogram Germe
def histogram_hesapla(gri_goruntu):
    hist = np.zeros(256, dtype=np.int64)
    for i in range(256):
        hist[i] = (gri_goruntu == i).sum()
    return hist

def histogram_germe(gri_goruntu):
    min_val = int(gri_goruntu.min())
    max_val = int(gri_goruntu.max())
    
    # Formül: (piksel - min) / (max - min) * 255
    if max_val != min_val:
        gerilmis = (gri_goruntu.astype(np.float64) - min_val) / (max_val - min_val) * 255.0
        return np.clip(gerilmis, 0, 255).astype(np.uint8)
    return gri_goruntu.copy()

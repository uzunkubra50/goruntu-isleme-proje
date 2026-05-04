# -*- coding: utf-8 -*-
import numpy as np
import math

# 4.3 - Görüntü Döndürme (Inverse Mapping)
def goruntu_dondur(goruntu, aci_derece, arka_plan=0):
    H, W = goruntu.shape[:2]
    t = math.radians(aci_derece)
    cos_t, sin_t = math.cos(t), math.sin(t)
    cx, cy = W / 2.0, H / 2.0

    cikti = np.full_like(goruntu, arka_plan)
    out_y, out_x = np.mgrid[0:H, 0:W]

    # Hedef pikselden kaynak piksele ters dönüşüm yapıyoruz (boşluk kalmaması için)
    dx, dy = out_x - cx, out_y - cy
    src_x = cos_t * dx + sin_t * dy + cx
    src_y = -sin_t * dx + cos_t * dy + cy

    src_xi, src_yi = np.round(src_x).astype(np.int32), np.round(src_y).astype(np.int32)
    
    # Görüntü sınırları içinde kalan pikselleri kopyalıyoruz
    gecerli = (src_xi >= 0) & (src_xi < W) & (src_yi >= 0) & (src_yi < H)
    cikti[gecerli] = goruntu[src_yi[gecerli], src_xi[gecerli]]
    return cikti

# 4.4 - Görüntü Kırpma
def goruntu_kirp(goruntu, x1, y1, x2, y2):
    # NumPy slicing (dilimleme) ile istenen bölgeyi alıyoruz
    H, W = goruntu.shape[:2]
    x1, y1 = max(0, min(x1, W-1)), max(0, min(y1, H-1))
    x2, y2 = max(0, min(x2, W)), max(0, min(y2, H))
    return goruntu[y1:y2, x1:x2].copy()

# 4.5 - Ölçekleme (Nearest Neighbor)
def goruntu_olcekle(goruntu, olcek_x, olcek_y=None):
    if olcek_y is None: olcek_y = olcek_x
    src_H, src_W = goruntu.shape[:2]
    out_W, out_H = max(1, int(src_W * olcek_x)), max(1, int(src_H * olcek_y))

    # Yeni koordinatları eski koordinatlara oranlayarak en yakın pikseli buluyoruz
    out_y, out_x = np.mgrid[0:out_H, 0:out_W]
    src_xi = np.clip((out_x / olcek_x).astype(np.int32), 0, src_W-1)
    src_yi = np.clip((out_y / olcek_y).astype(np.int32), 0, src_H-1)
    return goruntu[src_yi, src_xi].astype(np.uint8)

def goruntu_yakinlastir(goruntu, olcek):
    buyuk = goruntu_olcekle(goruntu, olcek)
    H_b, W_b = buyuk.shape[:2]
    H_o, W_o = goruntu.shape[:2]
    y1, x1 = (H_b - H_o)//2, (W_b - W_o)//2
    return buyuk[y1:y1+H_o, x1:x1+W_o].copy()

def goruntu_uzaklastir(goruntu, olcek):
    kucuk = goruntu_olcekle(goruntu, olcek)
    H_k, W_k = kucuk.shape[:2]
    H_o, W_o = goruntu.shape[:2]
    zemin = np.zeros_like(goruntu)
    y1, x1 = (H_o - H_k)//2, (W_o - W_k)//2
    zemin[y1:y1+H_k, x1:x1+W_k] = kucuk
    return zemin

# 4.8 - Aritmetik İşlemler
def goruntu_topla(img1, img2):
    # Taşmaları önlemek için int32'ye çekip sonra 0-255 arasına clip yapıyoruz
    return np.clip(img1.astype(np.int32) + img2.astype(np.int32), 0, 255).astype(np.uint8)

def goruntu_carp(img1, img2):
    return np.clip((img1.astype(np.float64) * img2.astype(np.float64)) / 255.0, 0, 255).astype(np.uint8)

def goruntu_fark(img1, img2):
    return np.clip(np.abs(img1.astype(np.int32) - img2.astype(np.int32)), 0, 255).astype(np.uint8)

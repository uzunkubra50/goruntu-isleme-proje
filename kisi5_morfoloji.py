# -*- coding: utf-8 -*-
import numpy as np

# 4.15 - Morfolojik İşlemler (Genişleme, Aşınma, Açma, Kapama)

def dilation(goruntu, kernel_size=3):
    # Binary'ye çevirip (127 eşik) h, w alıyoruz
    img = (goruntu > 127).astype(np.uint8)
    h, w = img.shape
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    cikti = np.zeros_like(img)
    
    # GENİŞLEME: Pencerede tek bir beyaz (1) varsa merkez beyaz olur
    for i in range(h):
        for j in range(w):
            pencere = padded[i:i+kernel_size, j:j+kernel_size]
            if np.any(pencere == 1):
                cikti[i, j] = 1
    return (cikti * 255).astype(np.uint8)

def erosion(goruntu, kernel_size=3):
    img = (goruntu > 127).astype(np.uint8)
    h, w = img.shape
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    cikti = np.zeros_like(img)
    
    # AŞINDIRMA: Sadece penceredeki TÜM pikseller beyaz (1) ise merkez beyaz kalır
    for i in range(h):
        for j in range(w):
            pencere = padded[i:i+kernel_size, j:j+kernel_size]
            if np.all(pencere == 1):
                cikti[i, j] = 1
    return (cikti * 255).astype(np.uint8)

def opening(goruntu, kernel_size=3):
    # Önce aşındırıp sonra genişletiyoruz (Gürültü siler)
    return dilation(erosion(goruntu, kernel_size), kernel_size)

def closing(goruntu, kernel_size=3):
    # Önce genişletip sonra aşındırıyoruz (Boşluk doldurur)
    return erosion(dilation(goruntu, kernel_size), kernel_size)

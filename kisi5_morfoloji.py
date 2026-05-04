# -*- coding: utf-8 -*-
"""
Kişi 5 - Morfolojik İşlemler Modülü
-----------------------------------
Bu dosya Dilation, Erosion, Opening ve Closing işlemlerini 
OpenCV kullanmadan, sadece NumPy ile yapar.
"""

import numpy as np

# Genişleme işlemi için yardımcı fonksiyon
def dilation(goruntu, kernel_size=3):
    # Görüntüyü 0 ve 1'lere çeviriyoruz
    img = (goruntu > 127).astype(np.uint8)
    h, w = img.shape
    
    # Padding (kenar doldurma) hesaplama
    pad = kernel_size // 2
    # Kenarları 0 ile dolduruyoruz ki kernel taşmasın
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    
    # Boş bir çıktı görüntüsü oluştur
    cikti = np.zeros_like(img)
    
    # Tüm pikseller üzerinde geziniyoruz
    for i in range(h):
        for j in range(w):
            # Kernel boyutunda bir pencere alıyoruz
            pencere = padded_img[i:i+kernel_size, j:j+kernel_size]
            
            # GENİŞLEME KURALI: Eğer pencerede en az bir tane beyaz (1) varsa merkez beyaz olur
            if np.any(pencere == 1):
                cikti[i, j] = 1
                
    # Tekrar 0-255 formatına çevirip döndür
    return (cikti * 255).astype(np.uint8)

# Aşındırma işlemi
def erosion(goruntu, kernel_size=3):
    img = (goruntu > 127).astype(np.uint8)
    h, w = img.shape
    pad = kernel_size // 2
    
    # Erosion'da kenarları 1 ile doldurmak nesneyi korumak için daha mantıklı olabilir 
    # ama genel kullanımda 0 (siyah) tercih edilir
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    cikti = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            pencere = padded_img[i:i+kernel_size, j:j+kernel_size]
            
            # AŞINDIRMA KURALI: Sadece penceredeki TÜM pikseller beyaz (1) ise merkez beyaz kalır
            if np.all(pencere == 1):
                cikti[i, j] = 1
                
    return (cikti * 255).astype(np.uint8)

# Açma işlemi (Erosion -> Dilation)
def opening(goruntu, kernel_size=3):
    # Önce aşındırıyoruz ki gürültüler gitsin
    gecici = erosion(goruntu, kernel_size)
    # Sonra genişletiyoruz ki nesne eski boyutuna dönsün
    sonuc = dilation(gecici, kernel_size)
    return sonuc

# Kapama işlemi (Dilation -> Erosion)
def closing(goruntu, kernel_size=3):
    # Önce genişletiyoruz ki boşluklar dolsun
    gecici = dilation(goruntu, kernel_size)
    # Sonra aşındırıyoruz ki nesne fazla büyümesin
    sonuc = erosion(gecici, kernel_size)
    return sonuc

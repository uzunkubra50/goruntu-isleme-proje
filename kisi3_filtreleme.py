# -*- coding: utf-8 -*-
import numpy as np

# 4.9 - Parlaklık ve Kontrast
def parlaklik_kontrast_ayari(image, alpha=1.0, beta=0):
    # Formül: alpha * piksel + beta
    sonuc = alpha * image.astype(np.int32) + beta
    return np.clip(sonuc, 0, 255).astype(np.uint8)

# 4.10 - Konvolüsyon ve Gauss Filtresi
def gauss_kernel_olustur(kernel_boyutu, sigma):
    # 2D Gauss formülünü uyguluyoruz: G(x,y) = exp(-(x^2+y^2)/(2*sigma^2))
    yari = kernel_boyutu // 2
    kernel = np.zeros((kernel_boyutu, kernel_boyutu))
    for i in range(kernel_boyutu):
        for j in range(kernel_boyutu):
            x, y = j - yari, i - yari
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum() # Normalizasyon: toplamı 1 yapıyoruz

def konvolusyon_uygula(image, kernel):
    # Sıfırdan manuel konvolüsyon: sliding window mantığı
    h, w = image.shape[:2]
    kh, kw = kernel.shape
    pad = kh // 2
    
    # Kenar pikselleri için görüntüyü sıfırla çevreliyoruz (padding)
    padded = np.pad(image, ((pad, pad), (pad, pad), (0,0)) if image.ndim==3 else ((pad,pad),(pad,pad)), mode='constant')
    cikti = np.zeros_like(image, dtype=np.float64)

    if image.ndim == 3: # Renkli resim
        for k in range(3):
            for i in range(h):
                for j in range(w):
                    bolge = padded[i:i+kh, j:j+kw, k]
                    cikti[i, j, k] = np.sum(bolge * kernel)
    else: # Gri resim
        for i in range(h):
            for j in range(w):
                bolge = padded[i:i+kh, j:j+kw]
                cikti[i, j] = np.sum(bolge * kernel)
                
    return np.clip(cikti, 0, 255).astype(np.uint8)

def gauss_filtresi(image, kernel_boyutu=5, sigma=1.0):
    kernel = gauss_kernel_olustur(kernel_boyutu, sigma)
    return konvolusyon_uygula(image, kernel)

# 4.14 - Bulanıklaştırma
def mean_blur(image, kernel_boyutu=3):
    # Tüm elemanları 1/(k*k) olan bir kernel ile ortalama alıyoruz
    kernel = np.ones((kernel_boyutu, kernel_boyutu)) / (kernel_boyutu**2)
    return konvolusyon_uygula(image, kernel)

def gaussian_blur(image, kernel_boyutu=5, sigma=1.0):
    return gauss_filtresi(image, kernel_boyutu, sigma)
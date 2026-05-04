# -*- coding: utf-8 -*-
import numpy as np

def rgb_to_gray(image):
    # Standart gri dönüşüm katsayıları
    return (0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]).astype(np.uint8)

# 4.11 - Global ve Adaptif Eşikleme
def global_esikleme(gray, esik=127):
    return np.where(gray >= esik, 255, 0).astype(np.uint8)

def adaptif_esikleme(gray, pencere_boyutu=11, C=5):
    h, w = gray.shape
    pad = pencere_boyutu // 2
    padded = np.pad(gray, pad, mode='constant')
    sonuc = np.zeros_like(gray)
    
    # Her pikselin etrafındaki komşuluğun ortalamasına bakıp eşik değerini dinamik belirliyoruz
    for i in range(h):
        for j in range(w):
            blok = padded[i:i+pencere_boyutu, j:j+pencere_boyutu]
            if gray[i, j] >= (np.mean(blok) - C):
                sonuc[i, j] = 255
    return sonuc

# 4.12 - Sobel Kenar Bulma
def sobel_kenar_bulma(gray, esik=50):
    # Yatay ve dikey kenar maskeleri
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy_mask = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Gradyanları konvolüsyon ile hesaplıyoruz
    from kisi3_filtreleme import konvolusyon_uygula
    gx = konvolusyon_uygula(gray, gx_mask).astype(np.float64)
    gy = konvolusyon_uygula(gray, gy_mask).astype(np.float64)
    
    # Gradyan büyüklüğü: sqrt(gx^2 + gy^2)
    mag = np.sqrt(gx**2 + gy**2)
    edges = np.where(mag >= esik, 255, 0).astype(np.uint8)
    return edges, mag, None

# 4.13 - Gürültü Ekleme ve Temizleme
def salt_pepper_gurultu_ekle(gray, gurultu_orani=0.05):
    sonuc = gray.copy()
    rastgele = np.random.random(gray.shape)
    sonuc[rastgele < gurultu_orani/2] = 0   # Pepper (Biber)
    sonuc[rastgele > 1 - gurultu_orani/2] = 255 # Salt (Tuz)
    return sonuc

def mean_filtre(gray, pencere_boyutu=3):
    from kisi3_filtreleme import mean_blur
    return mean_blur(gray, pencere_boyutu)

def median_filtre(gray, pencere_boyutu=3):
    h, w = gray.shape
    pad = pencere_boyutu // 2
    padded = np.pad(gray, pad, mode='constant')
    sonuc = np.zeros_like(gray)
    
    # Penceredeki değerleri sıralayıp ortadaki değeri (medyan) alıyoruz
    for i in range(h):
        for j in range(w):
            blok = padded[i:i+pencere_boyutu, j:j+pencere_boyutu]
            sonuc[i, j] = np.median(blok)
    return sonuc

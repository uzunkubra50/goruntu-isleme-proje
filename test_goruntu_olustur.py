# -*- coding: utf-8 -*-
"""
test_goruntu_olustur.py
------------------------
Bu script, projeyi test etmek için 'images/test.jpg' dosyasını oluşturur.
main.py çalıştırmadan önce SADECE BİR KEZ çalıştırın.

İÇERİK:
  - Büyük dikdörtgenler (morfoloji testi için)
  - İç boşluklu nesne (closing testi için)
  - Küçük gürültü pikselleri (opening testi için)
  - İnce çizgiler (erosion testi için)
"""

import numpy as np
import cv2
import os

# images klasörünü oluştur (yoksa)
os.makedirs('images', exist_ok=True)

# 400x600 siyah (boş) görüntü oluştur
img = np.zeros((400, 600), dtype=np.uint8)

# ---- Büyük dolu dikdörtgen (sol üst) ----
img[80:180, 80:250] = 255

# ---- Küçük gürültü dikdörtgeni (erosion/opening testi) ----
img[50:65, 350:380] = 255

# ---- İç boşluklu büyük kare (closing testi için) ----
img[200:330, 300:450] = 255
img[240:290, 340:410] = 0   # orta kısmı siyah bırak (delik)

# ---- İnce yatay çizgi (erosion testi için) ----
img[355:358, 50:540] = 255

# ---- İnce dikey çizgi ----
img[100:340, 148:151] = 255

# ---- Rastgele gürültü noktaları (opening testi) ----
rng = np.random.default_rng(42)
noise_rows = rng.integers(0, 400, 100)
noise_cols = rng.integers(0, 600, 100)
img[noise_rows, noise_cols] = 255

# Görüntüyü kaydet
cv2.imwrite('images/test.jpg', img)
print("[OK] Test goruntu olusturuldu: images/test.jpg")
print("     Boyut: {}x{} piksel".format(img.shape[1], img.shape[0]))

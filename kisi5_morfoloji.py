"""
kisi5_morfoloji.py
-------------------
Görüntü İşleme Dersi - Morfolojik İşlemler Modülü

Açıklama:
    Bu modül, temel morfolojik görüntü işleme operasyonlarını
    (Genişleme, Aşınma, Açma, Kapama) OpenCV'nin hazır fonksiyonları
    kullanmadan, sıfırdan NumPy ile implement eder.

    KURAL: cv2.erode, cv2.dilate, cv2.morphologyEx gibi hazır
    fonksiyonlar KESİNLİKLE KULLANILMAMAKTADIR.

    Tüm işlemler sliding window (kayan pencere) mantığı ile
    structuring element (yapısal eleman) kullanılarak yapılır.

Yazar: Kisi 5
Tarih: 2026
"""

import numpy as np


# ===========================================================================
# YARDIMCI FONKSİYON: Kare yapısal eleman (kernel) oluşturma
# ===========================================================================
def kare_kernel_olustur(kernel_size: int) -> np.ndarray:
    """
    Verilen boyutta tüm değerleri 1 olan kare yapısal eleman oluşturur.

    Parametreler:
        kernel_size (int): Kare kernelin kenar uzunluğu (örn. 3 -> 3x3 kernel)

    Döndürür:
        np.ndarray: kernel_size x kernel_size boyutunda, tüm elemanları 1
                    olan binary numpy dizisi (uint8).
    """
    # Tüm değerleri 1 olan kare matris oluştur (tam dolu disk yapısal eleman)
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)


# ===========================================================================
# FONKSİYON 1: GENİŞLEME (DILATION)
# ===========================================================================
def dilation(goruntu: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Morfolojik GENİŞLEME (Dilation) işlemi - sıfırdan implementasyon.

    Mantık:
        Yapısal eleman (kernel) görüntü üzerinde kayarken, kernelin
        kapsamındaki herhangi bir piksel 1 (beyaz) ise merkez pikseli
        de 1 (beyaz) yap. Yani LOCAL MAXIMUM alınır.

        Matematiksel ifade: (A ⊕ B)[i,j] = max(A[pencere])
        Eğer penceredeki piksellerin en az biri 1 ise → çıktı 1

    Parametreler:
        goruntu   (np.ndarray): İkili (binary) giriş görüntüsü (0 veya 255 değerli)
        kernel_size (int)     : Kare yapısal elemanın boyutu (varsayılan: 3)

    Döndürür:
        np.ndarray: Genişletilmiş ikili görüntü (uint8)
    """
    # Görüntüyü 0-1 aralığına normalize et (binary işlem için)
    # 255 olan beyaz pikseller 1, siyah pikseller 0 olacak
    ikilik_goruntu = (goruntu > 0).astype(np.uint8)

    # Görüntünün boyutlarını al
    yukseklik, genislik = ikilik_goruntu.shape[:2]

    # Kernel yarıçapını hesapla (padding miktarı)
    # Örnek: kernel_size=3 → pad=1, kernel_size=5 → pad=2
    pad = kernel_size // 2

    # Kenar piksellerinin kaybolmaması için görüntüye PADDING ekle
    # 'constant' modu ile kenarlar 0 değeriyle doldurulur (siyah)
    padded = np.pad(ikilik_goruntu, pad_width=pad, mode='constant', constant_values=0)

    # Çıktı görüntüsü için boş dizi oluştur (sıfırla başlat)
    cikti = np.zeros_like(ikilik_goruntu, dtype=np.uint8)

    # Kare yapısal elemanı oluştur
    kernel = kare_kernel_olustur(kernel_size)

    # -----------------------------------------------------------------------
    # SLIDING WINDOW (Kayan Pencere) Döngüsü
    # Her piksel üzerinde kernel boyutunda pencere kaydır
    # -----------------------------------------------------------------------
    for i in range(yukseklik):
        for j in range(genislik):
            # Mevcut pikselin etrafındaki kernel_size x kernel_size büyüklüğünde
            # pencereyi padded görüntüden çıkar
            pencere = padded[i:i + kernel_size, j:j + kernel_size]

            # Structural element (kernel) ile eleman bazlı çarpım yap
            # Kernel sadece dolu (1) alanlara bakmasını sağlar
            carpim = pencere * kernel

            # GENİŞLEME KURALI:
            # Penceredeki herhangi bir piksel 1 ise → çıktı pikseli 1 yap
            # Bu, np.max() veya np.any() ile yapılır
            if np.max(carpim) > 0:
                cikti[i, j] = 1

    # Çıktıyı tekrar 0-255 aralığına döndür (görüntü formatı için)
    return (cikti * 255).astype(np.uint8)


# ===========================================================================
# FONKSİYON 2: AŞINMA (EROSION)
# ===========================================================================
def erosion(goruntu: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Morfolojik AŞINMA (Erosion) işlemi - sıfırdan implementasyon.

    Mantık:
        Yapısal eleman (kernel) görüntü üzerinde kayarken, kernelin
        kapsamındaki TÜM pikseller 1 (beyaz) ise merkez pikseli
        1 (beyaz) yap; aksi hâlde 0 (siyah) yap. LOCAL MINIMUM alınır.

        Matematiksel ifade: (A ⊖ B)[i,j] = min(A[pencere])
        Eğer penceredeki TÜM pikseller 1 ise → çıktı 1
        Yeterli değilse → çıktı 0

    Parametreler:
        goruntu   (np.ndarray): İkili (binary) giriş görüntüsü (0 veya 255 değerli)
        kernel_size (int)     : Kare yapısal elemanın boyutu (varsayılan: 3)

    Döndürür:
        np.ndarray: Aşındırılmış ikili görüntü (uint8)
    """
    # Görüntüyü 0-1 aralığına normalize et
    ikilik_goruntu = (goruntu > 0).astype(np.uint8)

    yukseklik, genislik = ikilik_goruntu.shape[:2]

    # Kernel yarıçapını hesapla
    pad = kernel_size // 2

    # Kenarları 0 ile doldur (siyah padding — erosion'da kenar pikselleri erir)
    padded = np.pad(ikilik_goruntu, pad_width=pad, mode='constant', constant_values=0)

    # Çıktı dizisi
    cikti = np.zeros_like(ikilik_goruntu, dtype=np.uint8)

    # Kare yapısal elemanı oluştur
    kernel = kare_kernel_olustur(kernel_size)

    # -----------------------------------------------------------------------
    # SLIDING WINDOW (Kayan Pencere) Döngüsü
    # -----------------------------------------------------------------------
    for i in range(yukseklik):
        for j in range(genislik):
            # Mevcut piksel etrafındaki pencereyi çıkar
            pencere = padded[i:i + kernel_size, j:j + kernel_size]

            # Structural element ile çarpım
            carpim = pencere * kernel

            # AŞINMA KURALI:
            # Kernelin kapladığı alandaki TÜM pikseller 1 olmalıdır.
            # Kerneldeki 1 sayısı (toplam kernel alanı) ile
            # carpim toplamını karşılaştır.
            toplam_kernel = np.sum(kernel)   # Kerneldeki 1 sayısı
            toplam_carpim = np.sum(carpim)   # Penceredeki 1 sayısı

            # Eğer kernel altındaki tüm pikseller 1 ise çıktı 1
            if toplam_carpim == toplam_kernel:
                cikti[i, j] = 1

    return (cikti * 255).astype(np.uint8)


# ===========================================================================
# FONKSİYON 3: AÇMA (OPENING)
# ===========================================================================
def opening(goruntu: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Morfolojik AÇMA (Opening) işlemi - sıfırdan implementasyon.

    Mantık:
        Açma = Önce AŞINMA (Erosion), sonra GENİŞLEME (Dilation)
        Formül: A ∘ B = (A ⊖ B) ⊕ B

        Kullanım amacı:
        - Küçük beyaz gürültüleri (noise) temizler
        - Nesnelerin genel şekli ve büyüklüğü korunur
        - İnce bağlantıları keser

    Parametreler:
        goruntu   (np.ndarray): İkili (binary) giriş görüntüsü
        kernel_size (int)     : Kare yapısal elemanın boyutu (varsayılan: 3)

    Döndürür:
        np.ndarray: Açma uygulanmış ikili görüntü (uint8)
    """
    # ADIM 1: Önce erosion (aşınma) uygula → küçük beyaz bölgeler kaldırılır
    erosion_sonucu = erosion(goruntu, kernel_size)

    # ADIM 2: Ardından dilation (genişleme) uygula → nesneler orijinal boyutuna yaklaşır
    opening_sonucu = dilation(erosion_sonucu, kernel_size)

    return opening_sonucu


# ===========================================================================
# FONKSİYON 4: KAPAMA (CLOSING)
# ===========================================================================
def closing(goruntu: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Morfolojik KAPAMA (Closing) işlemi - sıfırdan implementasyon.

    Mantık:
        Kapama = Önce GENİŞLEME (Dilation), sonra AŞINMA (Erosion)
        Formül: A • B = (A ⊕ B) ⊖ B

        Kullanım amacı:
        - Küçük siyah delikleri (boşlukları) doldurur
        - Nesnelerdeki çatlakları ve boşlukları kapatır
        - Nesnelerin genel şekli ve büyüklüğü korunur

    Parametreler:
        goruntu   (np.ndarray): İkili (binary) giriş görüntüsü
        kernel_size (int)     : Kare yapısal elemanın boyutu (varsayılan: 3)

    Döndürür:
        np.ndarray: Kapama uygulanmış ikili görüntü (uint8)
    """
    # ADIM 1: Önce dilation (genişleme) uygula → boşluklar doldurulur
    dilation_sonucu = dilation(goruntu, kernel_size)

    # ADIM 2: Ardından erosion (aşınma) uygula → nesneler orijinal boyutuna yaklaşır
    closing_sonucu = erosion(dilation_sonucu, kernel_size)

    return closing_sonucu

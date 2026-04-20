# Görüntü İşleme Dersi — Grup 12

## Temel Görüntü İşleme Operasyonlarının Manuel İmplementasyonu

Bu projede temel görüntü işleme operasyonları, hazır kütüphane fonksiyonları kullanılmaksızın sıfırdan (from scratch) manuel olarak implement edilmektedir.

---

## Grup Üyeleri ve İş Bölümü

| Kişi | Sorumluluk | Dosya |
|------|-----------|-------|
| Kişi 1 | Gri Dönüşüm, Binary, Renk Uzayı (RGB→HSV), Histogram | `kisi1_temel.py` |
| Kişi 2 | Döndürme, Kırpma, Ölçekleme, Aritmetik İşlemler | `kisi2_geometrik.py` |
| Kişi 3 | Parlaklık/Kontrast, Konvolüsyon, Gauss, Bulanıklaştırma | `kisi3_filtreleme.py` |
| Kişi 4 | Eşikleme (Global & Adaptif), Sobel Kenar, Gürültü | `kisi4_kenar.py` |
| Kişi 5 | Morfolojik İşlemler, Ana İş Akışı, Entegrasyon, Arayüz | `kisi5_morfoloji.py` + `arayuz.py` |

---

## Kullanılan Teknolojiler

- **Python 3.10+** — ana geliştirme dili
- **NumPy** — tüm piksel işlemleri için (serbest)
- **OpenCV** — yalnızca `cv2.imread`, `cv2.imwrite`, `cv2.imdecode` (sınırlı)
- **Matplotlib** — görselleştirme ve histogram grafikleri
- **Pillow** — arayüz görsel desteği (ImageTk)

---

## Proje Yapısı

```
goruntu-isleme-proje/
├── images/              ← test görüntüleri (buraya koy)
├── outputs/             ← işlenmiş çıktılar (.gitignore'da)
├── arayuz.py            ← **ANA ARAYÜZ (Buradan çalıştırın)**
├── kisi1_temel.py
├── kisi2_geometrik.py
├── kisi3_filtreleme.py
├── kisi4_kenar.py
├── kisi5_morfoloji.py
├── main.py              ← terminal tabanlı ana akış
├── test_goruntu_olustur.py ← hızlı test için görsel üretici
├── README.md
└── .gitignore
```

---

## Kurulum

Gerekli tüm kütüphaneleri şu komutla yükleyebilirsiniz:

```bash
pip install numpy opencv-python matplotlib pillow
```

---

## Çalıştırma

Modern, karanlık tema destekli arayüzü başlatmak için:

```bash
python arayuz.py
```

*Not: Klasik terminal akışı için `python main.py` dosyasını da kullanabilirsiniz.*

---

## 🛠 Geliştiriciler İçin Önemli Notlar

### 1. Kendi Modülünüzü Arayüze Ekleme
Herkesin kendine ait bir sekmesi (Tab) `arayuz.py` içerisinde hazır durumdadır. Kendi fonksiyonlarınızı entegre etmek için:
1. `arayuz.py` dosyasının en üstünde kendi dosyanızı `import` edin.
2. `KisiXSekmesi` sınıfı içindeki placeholder (yer tutucu) kısmını kaldırıp kendi buton ve slider'larınızı ekleyin.
3. **Önemli:** Ağır işlemlerin arayüzü dondurmaması için Kişi 5'in yazdığı **Threading** (arka plan iş parçacığı) yapısını örnek alın.

### 2. Türkçe Karakter ve Dosya Yolu Sorunu
Görüntü yüklerken dosya yollarında Türkçe karakter (`ü, İ, ş, ğ` vb.) olması durumunda OpenCV hata verebilir. Bu projede bu sorun `np.fromfile` ve `cv2.imdecode` kullanılarak aşılmıştır. Lütfen kendi dosya işlemlerinizde `arayuz.py` içindeki `_goruntu_yukle` metodunu referans alın.

---

## ❓ Sorun Giderme (Sıkça Karşılaşılan Hatalar)

Eğer projeyi çalıştırırken hata alıyorsanız, lütfen aşağıdaki adımları kontrol edin:

### 1. "ModuleNotFoundError: No module named 'cv2'" Hatası
Bu hata OpenCV'nin yüklü olmadığını gösterir. Terminale şu komutu yazarak kütüphaneleri yeniden kurun:
```bash
pip install numpy opencv-python matplotlib pillow
```
*Not: Eğer bilgisayarınızda birden fazla Python sürümü varsa, kütüphaneyi doğru sürüme kurduğunuzdan emin olun (örn: `python -m pip install ...` veya `py -m pip install ...`).*

### 2. "Goruntu dosyasi bulunamadi: 'images/test.jpg'" Hatası
Arayüzü veya `main.py`'yi ilk kez çalıştırmadan önce mutlaka bir test görüntüsü oluşturmanız gerekir. Şu komutu çalıştırın:
```bash
python test_goruntu_olustur.py
```

### 3. Arayüz Açılıyor Ama Görüntü Yüklenmiyor
Dosya yollarında Türkçe karakter kullanmaktan kaçınmaya çalışın. Her ne kadar arayüzde bunu desteklemek için `imdecode` kullansak da, en güvenli yol görselleri `images/` klasörü içinde toplamak ve oradan açmaktır.

---

## Branch Kuralları

- Her kişi kendi branch'inde çalışır: `kisi1/temel`, `kisi2/geometrik` vb.
- `main` branch'e doğrudan push yapılmaz — Pull Request (PR) açılır.
- PR açmadan önce kendi modülünüzü izole test edin.

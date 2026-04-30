# Görüntü İşleme Projesi — Grup 12

Bu proje, temel görüntü işleme operasyonlarının (gri dönüşüm, geometrik işlemler, filtreleme, kenar bulma ve morfoloji) NumPy kullanılarak sıfırdan manuel olarak implement edildiği kapsamlı bir uygulamadır.

## 🚀 Öne Çıkan Özellikler
- **Tamamı Manuel:** Hazır kütüphane fonksiyonları (`cv2.blur`, `cv2.threshold` vb.) kullanılmadan yazılmış algoritmalar.
- **Modern Arayüz:** Tkinter tabanlı, karanlık mod destekli ve dinamik yan menülü kullanıcı dostu tasarım.
- **Türkçe Karakter Desteği:** Dosya yollarındaki Türkçe karakter sorununu çözen özel okuma mekanizması.
- **Threaded Yapı:** Görüntü işleme sırasında arayüzün donmasını engelleyen eşzamanlı işlem mimarisi.

## 🛠️ Teknolojiler
- **Python 3.10+**
- **NumPy** (Piksel işlemleri)
- **OpenCV** (Sadece okuma/yazma)
- **Matplotlib** (Görselleştirme)
- **Pillow** (Arayüz görsel desteği)

## 💻 Kurulum ve Çalıştırma

1. Bağımlılıkları yükleyin:
   ```bash
   pip install numpy opencv-python matplotlib pillow
   ```

2. Uygulamayı başlatın:
   ```bash
   python arayuz.py
   ```

## 📂 Proje Yapısı
- `arayuz.py`: Ana GUI uygulaması.
- `kisi1..kisi5_*.py`: Her grup üyesinin geliştirdiği özel modüller.
- `images/`: Test görselleri.
- `outputs/`: İşlenen görsellerin kaydedildiği dizin.

---
*Grup 12 tarafından Görüntü İşleme dersi kapsamında geliştirilmiştir.*

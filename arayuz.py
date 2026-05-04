# -*- coding: utf-8 -*-
"""
arayuz.py  —  Goruntu Isleme Dersi, Grup 12
---------------------------------------------
Tkinter tabanli ana arayuz. Her kisi kendi sekmesini buradan yonetir.

Kisi 1 : kisi1_temel.py      -> Gri, Binary, HSV, Histogram
Kisi 2 : kisi2_geometrik.py  -> Dondurme, Kirpma, Olcekleme, Aritmetik
Kisi 3 : kisi3_filtreleme.py -> Parlaklik, Konvolüsyon, Gauss, Bulanik
Kisi 4 : kisi4_kenar.py      -> Esikleme, Sobel, Gurultu
Kisi 5 : kisi5_morfoloji.py  -> Dilation, Erosion, Opening, Closing  [AKTIF]
"""

import os
import sys

# --------------------------------------------------------------------------
# TEKNİK NOT: TCL/TK YOL DÜZELTME
# Windows'ta kullanıcı adı Türkçe karakter (ü, ş, ç vb.) içerdiğinde 
# Tkinter kütüphaneleri bulunamayıp hata verebiliyor. 
# Bu fonksiyon Anaconda içindeki doğru yolları bulup sisteme tanıtır.
# --------------------------------------------------------------------------
def _tcl_yolu_duzenle():
    import glob
    # Python calistiricisinin bulundugu dizinden Anaconda kokunu tahmin et
    python_exe = sys.executable                      # orn: C:\Users\...\anaconda3\python.exe
    anaconda_kok = os.path.dirname(python_exe)       # orn: C:\Users\...\anaconda3

    # TCL 8.x surumu icin dizin ara
    tcl_pattern = os.path.join(anaconda_kok, "Library", "lib", "tcl8*")
    tk_pattern  = os.path.join(anaconda_kok, "Library", "lib", "tk8*")

    tcl_dizinler = sorted(glob.glob(tcl_pattern), reverse=True)
    tk_dizinler  = sorted(glob.glob(tk_pattern),  reverse=True)

    if tcl_dizinler:
        os.environ["TCL_LIBRARY"] = tcl_dizinler[0]
    if tk_dizinler:
        os.environ["TK_LIBRARY"] = tk_dizinler[0]

_tcl_yolu_duzenle()
# --------------------------------------------------------------------------

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk   # pip install pillow

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Kişi modülleri
from kisi5_morfoloji import dilation, erosion, opening, closing
from kisi1_temel_donusumler import gri_tonlama, binary_donusum, bgr_to_hsv, histogram_germe
from kisi2_geometrik import (goruntu_dondur, goruntu_kirp,
                              goruntu_olcekle, goruntu_topla,
                              goruntu_carp, goruntu_fark,
                              goruntu_yakinlastir, goruntu_uzaklastir)
from kisi3_filtreleme import parlaklik_kontrast_ayari, gauss_filtresi, mean_blur, gaussian_blur
from kisi4_goruntu_isleme import (global_esikleme, adaptif_esikleme, sobel_kenar_bulma,
                                  salt_pepper_gurultu_ekle, mean_filtre, median_filtre, rgb_to_gray)


# ---------------------------------------------------------------------------
# RENK ve STIL SABITLERI
# ---------------------------------------------------------------------------
BG_DARK    = "#1e1e2e"   # Ana arka plan (koyu lacivert)
BG_PANEL   = "#2a2a3d"   # Panel/kart arka plani
BG_CARD    = "#313149"   # Kart arka plani
ACCENT     = "#7c6af7"   # Mor vurgu rengi
ACCENT2    = "#56cfb2"   # Turkuaz ikincil vurgu
TEXT_MAIN  = "#e0e0f0"   # Ana metin
TEXT_DIM   = "#8888aa"   # Soluk metin
BORDER     = "#44445a"   # Kenar rengi
BTN_HOVER  = "#9880ff"   # Buton hover rengi
RED_WARN   = "#f07070"   # Uyari kirmizi
SUCCESS    = "#56cfb2"   # Basari yesili

FONT_TITLE  = ("Segoe UI", 18, "bold")
FONT_HEAD   = ("Segoe UI", 12, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)


# ===========================================================================
# YARDIMCI: Gorsel donusumleri
# ===========================================================================
def numpy_to_photoimage(arr: np.ndarray, max_w: int = 420, max_h: int = 340) -> ImageTk.PhotoImage:
    """NumPy dizisini tkinter PhotoImage'e cevirir (boyutu kisaltir)."""
    if arr is None:
        return None
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr.astype(np.uint8), mode='L')
    else:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb.astype(np.uint8))
    # Orani koru, max boyuta sigdir
    pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


def goruntu_gri_yap(bgr: np.ndarray) -> np.ndarray:
    """BGR goruntuyu NumPy ile gri tona cevir (cv2.cvtColor yasak degil ama kurala uygun)."""
    return (
        0.2989 * bgr[:, :, 2] +
        0.5870 * bgr[:, :, 1] +
        0.1140 * bgr[:, :, 0]
    ).astype(np.uint8)


def binary_yap(gri: np.ndarray, esik: int = 127) -> np.ndarray:
    return np.where(gri > esik, 255, 0).astype(np.uint8)


# ===========================================================================
# OZEL WIDGET: Hover efektli buton
# ===========================================================================
class HoverButton(tk.Button):
    def __init__(self, master, **kw):
        self._bg_normal = kw.get('bg', ACCENT)
        self._bg_hover  = kw.pop('hover_bg', BTN_HOVER)
        super().__init__(master, **kw)
        self.bind('<Enter>', lambda e: self.config(bg=self._bg_hover))
        self.bind('<Leave>', lambda e: self.config(bg=self._bg_normal))


# ===========================================================================
# SEKME TABANLI SINIF
# ===========================================================================
class SekmeBaz(ttk.Frame):
    """Her kisi sekmesi bu siniftan tureyecek."""

    def __init__(self, parent, goruntu_geri_donus_cb, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style='Dark.TFrame')
        self.goruntu_callback = goruntu_geri_donus_cb  # Ana pencereye goruntu gonder
        self._aktif_goruntu: np.ndarray = None         # Sekmede aktif goruntu

    def goruntu_ayarla(self, bgr_goruntu: np.ndarray):
        """Ana pencereden goruntu gelince cagrilir. Alt sinif override edecek."""
        self._aktif_goruntu = bgr_goruntu

    def _placeholder_goster(self, parent_frame: tk.Frame, kisi_no: int, aciklama: str):
        """Henuz hazir olmayan sekmeler icin gosterge kutu."""
        kart = tk.Frame(parent_frame, bg=BG_CARD, bd=0, highlightthickness=1,
                        highlightbackground=BORDER)
        kart.pack(expand=True, fill='both', padx=30, pady=30)

        tk.Label(kart, text="Kisi {}".format(kisi_no),
                 font=("Segoe UI", 36, "bold"), fg=ACCENT, bg=BG_CARD).pack(pady=(40, 5))

        tk.Label(kart, text=aciklama,
                 font=FONT_HEAD, fg=TEXT_MAIN, bg=BG_CARD).pack(pady=5)

        tk.Label(kart,
                 text="Bu sekme henuz hazirlanmamistir.\n"
                      "Ilgili kisi kendi modulunü buraya ekleyecektir.",
                 font=FONT_BODY, fg=TEXT_DIM, bg=BG_CARD, justify='center').pack(pady=10)

        tk.Label(kart,
                 text="Eklemek icin:\n"
                      "1. kisiX_modul.py dosyani projeye ekle\n"
                      "2. Ust kisideki import satirinin yorumunu kaldir\n"
                      "3. apply_islemler() fonksiyonunu doldur",
                 font=FONT_MONO, fg=ACCENT2, bg=BG_CARD, justify='left').pack(pady=(5, 40))


# ===========================================================================
# KİŞİ 1 SEKMESİ — Gri, Binary, HSV, Histogram [AKTIF]
# ===========================================================================
class Kisi1Sekmesi(SekmeBaz):
    def __init__(self, parent, cb, **kw):
        super().__init__(parent, cb, **kw)
        self._kart_imgbox = {}
        self._olustur_arayuz()

    def _olustur_arayuz(self):
        # ---- UST BAR ----
        ust = tk.Frame(self, bg=BG_PANEL, pady=10)
        ust.pack(fill='x')
        tk.Label(ust, text="Temel Dönüşümler", font=FONT_TITLE, fg=ACCENT, bg=BG_PANEL).pack(side='left', padx=20)

        # ---- KONTROL PANELİ ----
        kontrol = tk.Frame(self, bg=BG_CARD, bd=0, highlightthickness=1, highlightbackground=BORDER)
        kontrol.pack(fill='x', padx=15, pady=(8, 4))

        tk.Label(kontrol, text="Binary Eşik Değeri:", font=FONT_BODY, fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=0, padx=(15,5), pady=10)
        
        self._esik_var = tk.IntVar(value=127)
        slider = tk.Scale(kontrol, from_=0, to=255, resolution=1, orient='horizontal', variable=self._esik_var,
                          bg=BG_CARD, fg=TEXT_MAIN, troughcolor=BG_DARK, highlightthickness=0, bd=0, length=160)
        slider.grid(row=0, column=1, padx=5)

        HoverButton(kontrol, text="  Uygula  ", font=FONT_HEAD, bg=ACCENT, fg='white', relief='flat', cursor='hand2',
                    hover_bg=BTN_HOVER, padx=16, pady=6, command=self._uygula).grid(row=0, column=2, padx=30)

        # ---- GÖRÜNTÜ GRID (2 x 3) ----
        self._grid_frame = tk.Frame(self, bg=BG_DARK)
        self._grid_frame.pack(expand=True, fill='both', padx=15, pady=8)

        islemler = [
            ("Orijinal (BGR)", "orijinal"),
            ("1. Gri Tonlama", "gri"),
            ("2. Binary Dönüşüm", "binary"),
            ("3. RGB -> HSV", "hsv"),
            ("4. Histogram Germe", "hist")
        ]

        for idx, (baslik, anahtar) in enumerate(islemler):
            satir, sutun = divmod(idx, 3)
            kart = tk.Frame(self._grid_frame, bg=BG_CARD, bd=0, highlightthickness=1, highlightbackground=BORDER)
            kart.grid(row=satir, column=sutun, padx=8, pady=8, sticky='nsew')
            self._grid_frame.columnconfigure(sutun, weight=1)
            self._grid_frame.rowconfigure(satir, weight=1)

            tk.Label(kart, text=baslik, font=FONT_BODY, fg=ACCENT if idx==0 else TEXT_MAIN, bg=BG_CARD, pady=6).pack()
            imgbox = tk.Label(kart, bg=BG_DARK, text="Görüntü bekleniyor...", fg=TEXT_DIM, font=FONT_SMALL)
            imgbox.pack(expand=True, fill='both', padx=6, pady=(0, 6))
            self._kart_imgbox[anahtar] = imgbox

    def goruntu_ayarla(self, bgr: np.ndarray):
        super().goruntu_ayarla(bgr)
        self._guncelle_imgbox('orijinal', bgr)
        for k in ('gri', 'binary', 'hsv', 'hist'):
            self._kart_imgbox[k].config(image='', text="Uygula'ya basın")
            self._kart_imgbox[k].image = None

    def _uygula(self):
        if self._aktif_goruntu is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir görüntü yükleyin!")
            return

        esik = self._esik_var.get()
        bgr_kopya = self._aktif_goruntu.copy()

        import threading
        def _hesapla():
            try:
                # Kişi 1 Fonksiyonlarını Çağırıyoruz
                gri_img  = gri_tonlama(bgr_kopya)
                bin_img  = binary_donusum(gri_img, threshold=esik)
                hsv_img  = bgr_to_hsv(bgr_kopya)
                hist_img = histogram_germe(gri_img)

                sonuclar = {'orijinal': bgr_kopya, 'gri': gri_img, 'binary': bin_img, 'hsv': hsv_img, 'hist': hist_img}
                self.after(0, lambda: self._hesaplama_bitti(sonuclar))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Hata", str(exc)))

        threading.Thread(target=_hesapla, daemon=True).start()

    def _hesaplama_bitti(self, sonuclar):
        for k, arr in sonuclar.items():
            self._guncelle_imgbox(k, arr)

    def _guncelle_imgbox(self, anahtar: str, arr: np.ndarray):
        box = self._kart_imgbox.get(anahtar)
        if box is None: return
        photo = numpy_to_photoimage(arr, max_w=320, max_h=220)
        box.config(image=photo, text='')
        box.image = photo


# ===========================================================================
# KİŞİ 2 SEKMESİ — Dondurme, Kirpma, Olcekleme (PLACEHOLDER)
# ===========================================================================
class Kisi2Sekmesi(SekmeBaz):
    """
    Kisi 2 — Geometrik Islemler
    Dondurme | Kirpma | Olcekleme | Aritmetik Islemler
    """
 
    def __init__(self, parent, cb, **kw):
        super().__init__(parent, cb, **kw)
        self._sonuclar = {}
        self._kart_imgbox = {}
        self._olustur_arayuz()
 
    # ------------------------------------------------------------------
    def _olustur_arayuz(self):
        # ---- UST BAR ----
        ust = tk.Frame(self, bg=BG_PANEL, pady=10)
        ust.pack(fill='x')
        tk.Label(ust, text="Geometrik Islemler",
                 font=FONT_TITLE, fg=ACCENT, bg=BG_PANEL).pack(side='left', padx=20)
        tk.Label(ust, text="Kisi 2 — Dondurme | Kirpma | Olcekleme | Aritmetik",
                 font=FONT_BODY, fg=TEXT_DIM, bg=BG_PANEL).pack(side='left', padx=5)
 
        # ---- KONTROL PANELİ ----
        kontrol = tk.Frame(self, bg=BG_CARD, bd=0,
                           highlightthickness=1, highlightbackground=BORDER)
        kontrol.pack(fill='x', padx=15, pady=(8, 4))
 
        # -- Dondurme acisi --
        tk.Label(kontrol, text="Dondurme (°):", font=FONT_BODY,
                 fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=0, padx=(15,4), pady=10)
        self._aci_var = tk.IntVar(value=45)
        self._aci_lbl = tk.Label(kontrol, text="45°", font=FONT_HEAD,
                                  fg=ACCENT, bg=BG_CARD, width=5)
        self._aci_lbl.grid(row=0, column=2, padx=4)
        tk.Scale(kontrol, from_=-180, to=180, resolution=1,
                 orient='horizontal', variable=self._aci_var,
                 bg=BG_CARD, fg=TEXT_MAIN, troughcolor=BG_DARK,
                 highlightthickness=0, bd=0, sliderrelief='flat',
                 activebackground=ACCENT, length=140,
                 command=lambda v: self._aci_lbl.config(
                     text="{}°".format(v))).grid(row=0, column=1, padx=4)
 
        # -- Olcek faktoru --
        tk.Label(kontrol, text="Olcek (x):", font=FONT_BODY,
                 fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=3, padx=(20,4))
        self._olcek_var = tk.DoubleVar(value=1.5)
        self._olcek_lbl = tk.Label(kontrol, text="1.50x", font=FONT_HEAD,
                                    fg=ACCENT2, bg=BG_CARD, width=6)
        self._olcek_lbl.grid(row=0, column=5, padx=4)
        tk.Scale(kontrol, from_=0.1, to=4.0, resolution=0.1,
                 orient='horizontal', variable=self._olcek_var,
                 bg=BG_CARD, fg=TEXT_MAIN, troughcolor=BG_DARK,
                 highlightthickness=0, bd=0, sliderrelief='flat',
                 activebackground=ACCENT2, length=130,
                 command=lambda v: self._olcek_lbl.config(
                     text="{:.2f}x".format(float(v)))).grid(row=0, column=4, padx=4)
 
        # -- Uygula + Kaydet --
        self._buton_uygula = HoverButton(
            kontrol, text=" Uygula ", font=FONT_HEAD,
            bg=ACCENT, fg='white', relief='flat', cursor='hand2',
            hover_bg=BTN_HOVER, padx=16, pady=6,
            command=self._uygula)
        self._buton_uygula.grid(row=0, column=6, padx=(25, 8))
 
        HoverButton(kontrol, text=" Kaydet ", font=FONT_HEAD,
                    bg=BG_DARK, fg=ACCENT2, relief='flat', cursor='hand2',
                    hover_bg='#252535', padx=16, pady=6,
                    command=self._kaydet).grid(row=0, column=7, padx=(0, 15))
 
        # -- Kirpma ayarlari (ikinci satir) --
        tk.Label(kontrol, text="Kirpma (x1,y1,x2,y2):", font=FONT_BODY,
                 fg=TEXT_MAIN, bg=BG_CARD).grid(row=1, column=0, padx=(15,4), pady=(0,8))
 
        self._kirp_vars = []
        etiketler = ["x1", "y1", "x2", "y2"]
        varsayilanlar = [50, 30, 250, 180]
        for i, (et, vy) in enumerate(zip(etiketler, varsayilanlar)):
            tk.Label(kontrol, text=et+":", font=FONT_SMALL,
                     fg=TEXT_DIM, bg=BG_CARD).grid(row=1, column=1+i*2, padx=(8,2))
            var = tk.IntVar(value=vy)
            tk.Spinbox(kontrol, from_=0, to=9999, textvariable=var,
                       width=5, font=FONT_SMALL,
                       bg=BG_DARK, fg=TEXT_MAIN,
                       buttonbackground=BG_CARD,
                       relief='flat').grid(row=1, column=2+i*2, padx=(0,4))
            self._kirp_vars.append(var)
 
        # ---- GÖRÜNTÜ GRID (2 x 4) ----
        self._grid_frame = tk.Frame(self, bg=BG_DARK)
        self._grid_frame.pack(expand=True, fill='both', padx=15, pady=8)
 
        islemler = [
            ("Orijinal (BGR)",         "orijinal"),
            ("4.3 Dondurme",           "dondurme"),
            ("4.4 Kirpma",             "kirpma"),
            ("4.5 Yakinlastirma",      "buyut"),
            ("4.5 Uzaklastirma",       "kucult"),
            ("4.8 Toplama",            "toplama"),
            ("4.8 Carpma",             "carpma"),
            ("4.8 Fark",               "fark"),
        ]
 
        for idx, (baslik, anahtar) in enumerate(islemler):
            satir, sutun = divmod(idx, 4)
            kart = tk.Frame(self._grid_frame, bg=BG_CARD, bd=0,
                            highlightthickness=1, highlightbackground=BORDER)
            kart.grid(row=satir, column=sutun, padx=7, pady=7, sticky='nsew')
            self._grid_frame.columnconfigure(sutun, weight=1)
            self._grid_frame.rowconfigure(satir, weight=1)
 
            tk.Label(kart, text=baslik, font=FONT_BODY,
                     fg=ACCENT if idx == 0 else TEXT_MAIN,
                     bg=BG_CARD, pady=5).pack()
            imgbox = tk.Label(kart, bg=BG_DARK,
                              text="Goruntu bekleniyor...",
                              fg=TEXT_DIM, font=FONT_SMALL,
                              relief='flat', width=28, height=9)
            imgbox.pack(expand=True, fill='both', padx=5, pady=(0, 5))
            self._kart_imgbox[anahtar] = imgbox
 
        # Durum cubugu
        self._durum_var = tk.StringVar(value="Goruntu yukleyin ve 'Uygula' tusuna basin.")
        tk.Label(self, textvariable=self._durum_var,
                 font=FONT_SMALL, fg=TEXT_DIM, bg=BG_DARK,
                 anchor='w', padx=15, pady=4).pack(fill='x', side='bottom')
 
    # ------------------------------------------------------------------
    def goruntu_ayarla(self, bgr: np.ndarray):
        super().goruntu_ayarla(bgr)
        self._sonuclar = {}
        self._guncelle_imgbox('orijinal', bgr)
        for k in list(self._kart_imgbox.keys()):
            if k != 'orijinal':
                self._kart_imgbox[k].config(image='', text="Uygula'ya basin")
                self._kart_imgbox[k].image = None
        self._durum("Goruntu yuklendi. Parametreleri ayarlayin ve 'Uygula'ya basin.")
 
    # ------------------------------------------------------------------
    def _uygula(self):
        if self._aktif_goruntu is None:
            messagebox.showwarning("Uyari", "Lutfen once bir goruntu yukleyin!")
            return
 
        self._buton_uygula.config(state='disabled', text=" Isleniyor... ")
        self._durum("Geometrik islemler hesaplaniyor...")
        self.update_idletasks()
 
        bgr_kopya   = self._aktif_goruntu.copy()
        aci         = self._aci_var.get()
        olcek       = self._olcek_var.get()
        x1, y1, x2, y2 = [v.get() for v in self._kirp_vars]
 
        import threading
 
        def _hesapla():
            try:
                # 4.3 Dondurme
                dondu = goruntu_dondur(bgr_kopya, aci_derece=aci)
 
                # 4.4 Kirpma
                kirp  = goruntu_kirp(bgr_kopya, x1, y1, x2, y2)
 
                # 4.5 Yakinlastirma / Uzaklastirma
                buyut  = goruntu_yakinlastir(bgr_kopya, olcek)
                kucult = goruntu_uzaklastir(bgr_kopya, 0.5)
 
                # 4.8 Aritmetik: orijinal + aydinlatma katmani
                katman = np.full_like(bgr_kopya, 60)           # +60 parlaklik katmani
                topl  = goruntu_topla(bgr_kopya, katman)
                carp  = goruntu_carp(bgr_kopya,
                                     (bgr_kopya * 0.6).astype(np.uint8))
                fark  = goruntu_fark(bgr_kopya, katman)
 
                sonuclar = {
                    'orijinal': bgr_kopya,
                    'dondurme': dondu,
                    'kirpma'  : kirp,
                    'buyut'   : buyut,
                    'kucult'  : kucult,
                    'toplama' : topl,
                    'carpma'  : carp,
                    'fark'    : fark,
                }
                self.after(0, lambda: self._hesaplama_bitti(sonuclar, aci, olcek))
            except Exception as exc:
                error_msg = str(exc)
                self.after(0, lambda: self._hata(error_msg))
 
        threading.Thread(target=_hesapla, daemon=True).start()
 
    # ------------------------------------------------------------------
    def _hesaplama_bitti(self, sonuclar, aci, olcek):
        self._sonuclar = sonuclar
        for k, arr in sonuclar.items():
            self._guncelle_imgbox(k, arr)
        self._durum(
            "Tamamlandi! Dondurme: {}°  |  Olcek: {:.2f}x  |  "
            "Goruntu: {}x{}".format(
                aci, olcek,
                sonuclar['orijinal'].shape[1],
                sonuclar['orijinal'].shape[0])
        )
        self._buton_uygula.config(state='normal', text=" Uygula ")
 
    # ------------------------------------------------------------------
    def _hata(self, mesaj):
        messagebox.showerror("Hata", "Islem hatasi:\\n\\n" + mesaj)
        self._durum("HATA: " + mesaj)
        self._buton_uygula.config(state='normal', text=" Uygula ")
 
    # ------------------------------------------------------------------
    def _guncelle_imgbox(self, anahtar, arr):
        box = self._kart_imgbox.get(anahtar)
        if box is None:
            return
        photo = numpy_to_photoimage(arr, max_w=280, max_h=200)
        box.config(image=photo, text='')
        box.image = photo
 
    # ------------------------------------------------------------------
    def _kaydet(self):
        if not self._sonuclar:
            messagebox.showinfo("Bilgi", "Once 'Uygula'ya basin!")
            return
        os.makedirs('outputs', exist_ok=True)
        for isim, arr in self._sonuclar.items():
            cv2.imwrite("outputs/kisi2_{}.jpg".format(isim), arr)
        messagebox.showinfo("Kaydedildi",
                            "Goruntüler 'outputs/' klasorune kaydedildi.")
        self._durum("Goruntüler 'outputs/' klasorune kaydedildi.")
 
    # ------------------------------------------------------------------
    def _durum(self, mesaj):
        self._durum_var.set(mesaj)
        self.update_idletasks()

 
if __name__ == "__main__":
    print("Bu dosya dogrudan calistirilmaz.")
    print("Icerigini arayuz.py'ye entegre etmek icin README talimatlarini izleyin.")
 


# ===========================================================================
# KİŞİ 3 SEKMESİ — Parlaklik, Konvolusyon, Gauss (PLACEHOLDER)
# ===========================================================================
class Kisi3Sekmesi(SekmeBaz):
    """
    Kisi 3 — Filtreleme & Konvolüsyon
    Parlaklik/Kontrast | Gauss Filtresi | Mean/Gauss Blur
    """
    def __init__(self, parent, cb, **kw):
        super().__init__(parent, cb, **kw)
        self._sonuclar = {}
        self._kart_imgbox = {}
        self._olustur_arayuz()

    def _olustur_arayuz(self):
        ust = tk.Frame(self, bg=BG_PANEL, pady=10)
        ust.pack(fill='x')
        tk.Label(ust, text="Filtreleme & Konvolüsyon", font=FONT_TITLE, fg=ACCENT, bg=BG_PANEL).pack(side='left', padx=20)

        kontrol = tk.Frame(self, bg=BG_CARD, bd=0, highlightthickness=1, highlightbackground=BORDER)
        kontrol.pack(fill='x', padx=15, pady=(8, 4))

        # Kontrast (Alpha)
        tk.Label(kontrol, text="Kontrast:", font=FONT_BODY, fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=0, padx=(15,5))
        self._alpha_var = tk.DoubleVar(value=1.0)
        tk.Scale(kontrol, from_=0.1, to=3.0, resolution=0.1, orient='horizontal', variable=self._alpha_var,
                 bg=BG_CARD, fg=TEXT_MAIN, length=120, highlightthickness=0).grid(row=0, column=1)

        # Parlaklık (Beta)
        tk.Label(kontrol, text="Parlaklık:", font=FONT_BODY, fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=2, padx=(15,5))
        self._beta_var = tk.IntVar(value=0)
        tk.Scale(kontrol, from_=-100, to=100, orient='horizontal', variable=self._beta_var,
                 bg=BG_CARD, fg=TEXT_MAIN, length=120, highlightthickness=0).grid(row=0, column=3)

        # Kernel Boyutu
        tk.Label(kontrol, text="Kernel:", font=FONT_BODY, fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=4, padx=(15,5))
        self._ksize_var = tk.IntVar(value=5)
        tk.Scale(kontrol, from_=3, to=15, resolution=2, orient='horizontal', variable=self._ksize_var,
                 bg=BG_CARD, fg=TEXT_MAIN, length=100, highlightthickness=0).grid(row=0, column=5)

        # Uygula Butonu
        self._btn = HoverButton(kontrol, text=" Uygula ", font=FONT_HEAD, bg=ACCENT, fg='white', relief='flat',
                                 padx=15, pady=5, command=self._uygula)
        self._btn.grid(row=0, column=6, padx=20, pady=10)

        # Görüntü Alanı
        self._grid_frame = tk.Frame(self, bg=BG_DARK)
        self._grid_frame.pack(expand=True, fill='both', padx=15, pady=8)

        islemler = [
            ("Orijinal", "orijinal"),
            ("Parlaklık/Kontrast", "pk"),
            ("Gauss Filtresi", "gauss"),
            ("Mean Blur", "mean"),
            ("Gaussian Blur", "gblur")
        ]

        for idx, (baslik, anahtar) in enumerate(islemler):
            satir, sutun = divmod(idx, 3)
            kart = tk.Frame(self._grid_frame, bg=BG_CARD, bd=0, highlightthickness=1, highlightbackground=BORDER)
            kart.grid(row=satir, column=sutun, padx=8, pady=8, sticky='nsew')
            self._grid_frame.columnconfigure(sutun, weight=1)
            self._grid_frame.rowconfigure(satir, weight=1)
            tk.Label(kart, text=baslik, font=FONT_BODY, fg=TEXT_MAIN, bg=BG_CARD, pady=5).pack()
            box = tk.Label(kart, bg=BG_DARK, text="Bekleniyor...")
            box.pack(expand=True, fill='both', padx=5, pady=5)
            self._kart_imgbox[anahtar] = box

    def goruntu_ayarla(self, bgr):
        super().goruntu_ayarla(bgr)
        self._guncelle_imgbox('orijinal', bgr)

    def _uygula(self):
        if self._aktif_goruntu is None: return
        self._btn.config(state='disabled', text="Isleniyor...")
        bgr = self._aktif_goruntu.copy()
        alpha = self._alpha_var.get()
        beta = self._beta_var.get()
        k = self._ksize_var.get()

        # TEKNİK NOT: THREADING KULLANIMI
        # Görüntü işleme döngüleri ağır olduğu için ana thread'i (arayüzü) dondurur.
        # İşlemi arka planda (Thread) çalıştırarak arayüzün akıcı kalmasını sağlıyoruz.
        import threading
        def _run():
            try:
                pk = parlaklik_kontrast_ayari(bgr, alpha=alpha, beta=beta)
                gri = goruntu_gri_yap(bgr)
                gauss = gauss_filtresi(gri, kernel_boyutu=k, sigma=1.0)
                mean = mean_blur(gri, kernel_boyutu=k)
                gblur = gaussian_blur(gri, kernel_boyutu=k, sigma=1.0)
                
                sonuclar = {'orijinal': bgr, 'pk': pk, 'gauss': gauss, 'mean': mean, 'gblur': gblur}
                # after() metodu ile sonuçları arayüz thread'ine güvenli bir şekilde geri gönderiyoruz
                self.after(0, lambda: self._bitti(sonuclar))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Hata", str(e)))
                self.after(0, lambda: self._btn.config(state='normal', text=" Uygula "))

        threading.Thread(target=_run, daemon=True).start()

    def _bitti(self, sonuclar):
        self._sonuclar = sonuclar
        for k, arr in sonuclar.items(): self._guncelle_imgbox(k, arr)
        self._btn.config(state='normal', text=" Uygula ")

    def _guncelle_imgbox(self, k, arr):
        box = self._kart_imgbox.get(k)
        if box:
            photo = numpy_to_photoimage(arr, max_w=320, max_h=220)
            box.config(image=photo, text='')
            box.image = photo

class Kisi4Sekmesi(SekmeBaz):
    """
    Kisi 4 — Görüntü İşleme
    Eşikleme | Sobel Kenar | Gürültü Ekleme & Temizleme
    """
    def __init__(self, parent, cb, **kw):
        super().__init__(parent, cb, **kw)
        self._sonuclar = {}
        self._kart_imgbox = {}
        self._olustur_arayuz()

    def _olustur_arayuz(self):
        ust = tk.Frame(self, bg=BG_PANEL, pady=10)
        ust.pack(fill='x')
        tk.Label(ust, text="Kenar & Eşikleme & Gürültü", font=FONT_TITLE, fg=ACCENT, bg=BG_PANEL).pack(side='left', padx=20)

        kontrol = tk.Frame(self, bg=BG_CARD, bd=0, highlightthickness=1, highlightbackground=BORDER)
        kontrol.pack(fill='x', padx=15, pady=(8, 4))

        # Eşikleme Değerleri
        tk.Label(kontrol, text="Eşik:", font=FONT_BODY, fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=0, padx=5)
        self._esik_var = tk.IntVar(value=127)
        tk.Scale(kontrol, from_=0, to=255, orient='horizontal', variable=self._esik_var, bg=BG_CARD, fg=TEXT_MAIN, length=100, highlightthickness=0).grid(row=0, column=1)

        tk.Label(kontrol, text="Sobel Eşik:", font=FONT_BODY, fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=2, padx=5)
        self._sobel_var = tk.IntVar(value=50)
        tk.Scale(kontrol, from_=0, to=200, orient='horizontal', variable=self._sobel_var, bg=BG_CARD, fg=TEXT_MAIN, length=100, highlightthickness=0).grid(row=0, column=3)

        tk.Label(kontrol, text="Gürültü %:", font=FONT_BODY, fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=4, padx=5)
        self._noise_var = tk.DoubleVar(value=0.05)
        tk.Scale(kontrol, from_=0, to=0.5, resolution=0.01, orient='horizontal', variable=self._noise_var, bg=BG_CARD, fg=TEXT_MAIN, length=100, highlightthickness=0).grid(row=0, column=5)

        self._btn = HoverButton(kontrol, text=" Uygula ", font=FONT_HEAD, bg=ACCENT, fg='white', relief='flat', padx=15, pady=5, command=self._uygula)
        self._btn.grid(row=0, column=6, padx=20, pady=10)

        # Görüntü Alanı
        self._grid_frame = tk.Frame(self, bg=BG_DARK)
        self._grid_frame.pack(expand=True, fill='both', padx=15, pady=8)

        islemler = [
            ("Gri Orijinal", "gri"),
            ("Global Eşik", "global"),
            ("Adaptif Eşik", "adaptif"),
            ("Sobel Kenar", "sobel"),
            ("S&P Gürültü", "noise"),
            ("Median Filtre", "median")
        ]

        for idx, (baslik, anahtar) in enumerate(islemler):
            satir, sutun = divmod(idx, 3)
            kart = tk.Frame(self._grid_frame, bg=BG_CARD, bd=0, highlightthickness=1, highlightbackground=BORDER)
            kart.grid(row=satir, column=sutun, padx=8, pady=8, sticky='nsew')
            self._grid_frame.columnconfigure(sutun, weight=1)
            self._grid_frame.rowconfigure(satir, weight=1)
            tk.Label(kart, text=baslik, font=FONT_BODY, fg=TEXT_MAIN, bg=BG_CARD, pady=5).pack()
            box = tk.Label(kart, bg=BG_DARK, text="Bekleniyor...")
            box.pack(expand=True, fill='both', padx=5, pady=5)
            self._kart_imgbox[anahtar] = box

    def goruntu_ayarla(self, bgr):
        super().goruntu_ayarla(bgr)
        gri = rgb_to_gray(bgr)
        self._guncelle_imgbox('gri', gri)

    def _uygula(self):
        if self._aktif_goruntu is None: return
        self._btn.config(state='disabled', text="Isleniyor...")
        bgr = self._aktif_goruntu.copy()
        esik = self._esik_var.get()
        sobel_esik = self._sobel_var.get()
        noise_rate = self._noise_var.get()

        import threading
        def _run():
            try:
                gri = rgb_to_gray(bgr)
                glob = global_esikleme(gri, esik=esik)
                adap = adaptif_esikleme(gri, pencere_boyutu=11, C=5)
                edges, _, _ = sobel_kenar_bulma(gri, esik=sobel_esik)
                noise = salt_pepper_gurultu_ekle(gri, gurultu_orani=noise_rate)
                med = median_filtre(noise, pencere_boyutu=3)
                
                sonuclar = {'gri': gri, 'global': glob, 'adaptif': adap, 'sobel': edges, 'noise': noise, 'median': med}
                self.after(0, lambda: self._bitti(sonuclar))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Hata", str(e)))
                self.after(0, lambda: self._btn.config(state='normal', text=" Uygula "))

        threading.Thread(target=_run, daemon=True).start()

    def _bitti(self, sonuclar):
        self._sonuclar = sonuclar
        for k, arr in sonuclar.items(): self._guncelle_imgbox(k, arr)
        self._btn.config(state='normal', text=" Uygula ")

    def _guncelle_imgbox(self, k, arr):
        box = self._kart_imgbox.get(k)
        if box:
            photo = numpy_to_photoimage(arr, max_w=320, max_h=220)
            box.config(image=photo, text='')
            box.image = photo


# ===========================================================================
# KİŞİ 5 SEKMESİ — Morfolojik Islemler  [AKTIF - TAM CALISIR]
# ===========================================================================
class Kisi5Sekmesi(SekmeBaz):

    def __init__(self, parent, cb, **kw):
        super().__init__(parent, cb, **kw)
        self._sonuclar = {}   # islem_adi -> numpy dizisi
        self._canvas_refs = {}
        self._olustur_arayuz()

    # ------------------------------------------------------------------
    def _olustur_arayuz(self):
        # ---- UST BAR: baslik + aciklama ----
        ust = tk.Frame(self, bg=BG_PANEL, pady=10)
        ust.pack(fill='x', padx=0)

        tk.Label(ust, text="Morfolojik Islemler",
                 font=FONT_TITLE, fg=ACCENT, bg=BG_PANEL).pack(side='left', padx=20)
        tk.Label(ust, text="Kisi 5  —  Dilation | Erosion | Opening | Closing",
                 font=FONT_BODY, fg=TEXT_DIM, bg=BG_PANEL).pack(side='left', padx=5)

        # ---- KONTROL PANELİ ----
        kontrol = tk.Frame(self, bg=BG_CARD, bd=0,
                           highlightthickness=1, highlightbackground=BORDER)
        kontrol.pack(fill='x', padx=15, pady=(8, 4))

        # Kernel slider
        tk.Label(kontrol, text="Kernel Boyutu:", font=FONT_BODY,
                 fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=0, padx=(15, 5), pady=10)

        self._kernel_var = tk.IntVar(value=5)
        self._kernel_lbl = tk.Label(kontrol, text="5x5", font=FONT_HEAD,
                                    fg=ACCENT, bg=BG_CARD, width=5)
        self._kernel_lbl.grid(row=0, column=2, padx=5)

        slider = tk.Scale(kontrol, from_=3, to=15, resolution=2,
                          orient='horizontal', variable=self._kernel_var,
                          bg=BG_CARD, fg=TEXT_MAIN, troughcolor=BG_DARK,
                          highlightthickness=0, bd=0, sliderrelief='flat',
                          activebackground=ACCENT, length=160,
                          command=lambda v: self._kernel_lbl.config(
                              text="{}x{}".format(v, v)))
        slider.grid(row=0, column=1, padx=5)

        # Esik slider
        tk.Label(kontrol, text="Binary Esik:", font=FONT_BODY,
                 fg=TEXT_MAIN, bg=BG_CARD).grid(row=0, column=3, padx=(25, 5))
        self._esik_var = tk.IntVar(value=127)
        self._esik_lbl = tk.Label(kontrol, text="127", font=FONT_HEAD,
                                  fg=ACCENT2, bg=BG_CARD, width=4)
        self._esik_lbl.grid(row=0, column=5, padx=5)

        slider2 = tk.Scale(kontrol, from_=0, to=255, resolution=1,
                           orient='horizontal', variable=self._esik_var,
                           bg=BG_CARD, fg=TEXT_MAIN, troughcolor=BG_DARK,
                           highlightthickness=0, bd=0, sliderrelief='flat',
                           activebackground=ACCENT2, length=160,
                           command=lambda v: self._esik_lbl.config(text=str(v)))
        slider2.grid(row=0, column=4, padx=5)

        # Calistir butonu
        self._buton_uygula = HoverButton(
            kontrol, text="  Uygula  ", font=FONT_HEAD,
            bg=ACCENT, fg='white', relief='flat', cursor='hand2',
            hover_bg=BTN_HOVER, padx=16, pady=6,
            command=self._uygula)
        self._buton_uygula.grid(row=0, column=6, padx=(30, 15))

        # Kaydet butonu
        HoverButton(kontrol, text="  Kaydet  ", font=FONT_HEAD,
                    bg=BG_DARK, fg=ACCENT2, relief='flat', cursor='hand2',
                    hover_bg='#252535', padx=16, pady=6,
                    command=self._kaydet).grid(row=0, column=7, padx=(0, 15))

        # ---- GÖRÜNTÜ GRID (2 x 3) ----
        self._grid_frame = tk.Frame(self, bg=BG_DARK)
        self._grid_frame.pack(expand=True, fill='both', padx=15, pady=8)

        self._kart_etiketler = {}
        self._kart_imgbox   = {}

        islemler = [
            ("Orijinal (Gri)", "orijinal"),
            ("Binary",         "binary"),
            ("Dilation\n(Genisleme)", "dilation"),
            ("Erosion\n(Asinma)",     "erosion"),
            ("Opening\n(Acma)",       "opening"),
            ("Closing\n(Kapama)",     "closing"),
        ]

        for idx, (baslik, anahtar) in enumerate(islemler):
            satir, sutun = divmod(idx, 3)
            kart = tk.Frame(self._grid_frame, bg=BG_CARD, bd=0,
                            highlightthickness=1, highlightbackground=BORDER)
            kart.grid(row=satir, column=sutun, padx=8, pady=8, sticky='nsew')
            self._grid_frame.columnconfigure(sutun, weight=1)
            self._grid_frame.rowconfigure(satir, weight=1)

            # Baslik
            tk.Label(kart, text=baslik, font=FONT_BODY, fg=ACCENT if idx == 0 else TEXT_MAIN,
                     bg=BG_CARD, pady=6).pack()

            # Goruntu kutu
            imgbox = tk.Label(kart, bg=BG_DARK, text="Goruntu bekleniyor...",
                              fg=TEXT_DIM, font=FONT_SMALL,
                              relief='flat', width=38, height=10)
            imgbox.pack(expand=True, fill='both', padx=6, pady=(0, 6))
            self._kart_imgbox[anahtar] = imgbox

        # ---- DURUM CUBUGU ----
        self._durum_var = tk.StringVar(value="Goruntu yukleyin ve 'Uygula' tusuna basin.")
        durum_bar = tk.Label(self, textvariable=self._durum_var,
                             font=FONT_SMALL, fg=TEXT_DIM, bg=BG_DARK,
                             anchor='w', padx=15, pady=4)
        durum_bar.pack(fill='x', side='bottom')

    # ------------------------------------------------------------------
    def goruntu_ayarla(self, bgr: np.ndarray):
        """Ana pencereden goruntu geldi."""
        super().goruntu_ayarla(bgr)
        self._sonuclar = {}
        # Sadece orijinali goster, hesaplama yapma
        gri = goruntu_gri_yap(bgr)
        self._guncelle_imgbox('orijinal', gri)
        # Diger kutulari temizle
        for k in ('binary', 'dilation', 'erosion', 'opening', 'closing'):
            self._kart_imgbox[k].config(image='', text="Uygula'ya basin")
            self._kart_imgbox[k].image = None
        self._durum("Goruntu yuklendi. Parametreleri secin ve 'Uygula'ya basin.")

    # ------------------------------------------------------------------
    def _uygula(self):
        """
        Morfolojik islemleri arka planda (thread) calistirir.
        Ana thread serbest kalir -- GUI donmaz / kapanmaz.
        """
        if self._aktif_goruntu is None:
            messagebox.showwarning("Uyari", "Lutfen once bir goruntu yukleyin!")
            return

        # Cift tiklamaya karsi butonu devre disi birak
        self._buton_uygula.config(state='disabled', text="  Isleniyor...  ")
        self._durum("Goruntu isleniyor, lutfen bekleyin...")
        self.update_idletasks()

        kernel    = self._kernel_var.get()
        esik      = self._esik_var.get()
        bgr_kopya = self._aktif_goruntu.copy()  # thread-safe kopya

        import threading

        def _hesapla():
            """Agir hesaplama: arka planda calisir."""
            try:
                gri    = goruntu_gri_yap(bgr_kopya)
                binary = binary_yap(gri, esik)
                dil    = dilation(binary, kernel_size=kernel)
                ero    = erosion(binary, kernel_size=kernel)
                ope    = opening(binary, kernel_size=kernel)
                clo    = closing(binary, kernel_size=kernel)

                sonuclar = {
                    'orijinal': gri,
                    'binary':   binary,
                    'dilation': dil,
                    'erosion':  ero,
                    'opening':  ope,
                    'closing':  clo,
                }
                # Sonuclari ana thread'e gonder (after() thread-safe'dir)
                self.after(0, lambda: self._hesaplama_bitti(sonuclar, kernel, esik))

            except Exception as exc:
                self.after(0, lambda: self._hesaplama_hatasi(str(exc)))

        threading.Thread(target=_hesapla, daemon=True).start()

    # ------------------------------------------------------------------
    def _hesaplama_bitti(self, sonuclar: dict, kernel: int, esik: int):
        """Arka plan thread bitince ana thread buraya cagrilir (after ile)."""
        self._sonuclar = sonuclar

        for k, arr in sonuclar.items():
            self._guncelle_imgbox(k, arr)

        gri = sonuclar.get('orijinal')
        self._durum(
            "Tamamlandi!  Kernel: {}x{}  |  Esik: {}  |  "
            "Goruntu: {}x{}".format(
                kernel, kernel, esik,
                gri.shape[1] if gri is not None else '?',
                gri.shape[0] if gri is not None else '?')
        )
        self._buton_uygula.config(state='normal', text="  Uygula  ")

    # ------------------------------------------------------------------
    def _hesaplama_hatasi(self, hata_mesaji: str):
        """Thread'den gelen hatayi goster."""
        messagebox.showerror("Islem Hatasi",
                             "Morfolojik islem sirasinda hata:\n\n" + hata_mesaji)
        self._durum("HATA: " + hata_mesaji)
        self._buton_uygula.config(state='normal', text="  Uygula  ")

    # ------------------------------------------------------------------
    def _guncelle_imgbox(self, anahtar: str, arr: np.ndarray):
        box = self._kart_imgbox.get(anahtar)
        if box is None:
            return
        photo = numpy_to_photoimage(arr, max_w=320, max_h=220)
        box.config(image=photo, text='')
        box.image = photo  # referansi tut

    # ------------------------------------------------------------------
    def _kaydet(self):
        """Tum sonuclari outputs/ klasorune kaydet."""
        if not self._sonuclar:
            messagebox.showinfo("Bilgi", "Once 'Uygula'ya basin!")
            return

        os.makedirs('outputs', exist_ok=True)
        for isim, arr in self._sonuclar.items():
            dosya = "outputs/kisi5_{}.jpg".format(isim)
            cv2.imwrite(dosya, arr)

        messagebox.showinfo("Kaydedildi",
                            "7 goruntu 'outputs/' klasorune kaydedildi.")
        self._durum("Goruntüler 'outputs/' klasorune kaydedildi.")

    # ------------------------------------------------------------------
    def _durum(self, mesaj: str):
        self._durum_var.set(mesaj)
        self.update_idletasks()


# ===========================================================================
# ANA PENCERE
# ===========================================================================
class AnaPencere(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Goruntu Isleme — Grup 12")
        self.configure(bg=BG_DARK)
        self.geometry("1280x780")
        self.minsize(900, 600)

        self._aktif_goruntu: np.ndarray = None
        self._stiller_ayarla()
        self._arayuz_olustur()

    # ------------------------------------------------------------------
    def _stiller_ayarla(self):
        stil = ttk.Style(self)
        stil.theme_use('clam')

        stil.configure('Dark.TFrame',   background=BG_DARK)
        stil.configure('Panel.TFrame',  background=BG_PANEL)

        # Notebook (sekme konteyner)
        stil.configure('Dark.TNotebook',
                        background=BG_DARK, borderwidth=0, tabmargins=0)
        stil.configure('Dark.TNotebook.Tab',
                        background=BG_PANEL, foreground=TEXT_DIM,
                        font=FONT_BODY, padding=[18, 8], borderwidth=0)
        stil.map('Dark.TNotebook.Tab',
                 background=[('selected', BG_CARD), ('active', BG_CARD)],
                 foreground=[('selected', ACCENT), ('active', TEXT_MAIN)])

    # ------------------------------------------------------------------
    def _arayuz_olustur(self):
        # ============================================================
        # UST BAR: logo + goruntu yukle
        # ============================================================
        ust_bar = tk.Frame(self, bg=BG_PANEL, height=56)
        ust_bar.pack(fill='x')
        ust_bar.pack_propagate(False)

        # Sol: proje basligi
        sol = tk.Frame(ust_bar, bg=BG_PANEL)
        sol.pack(side='left', fill='y', padx=15)

        tk.Label(sol, text="Goruntu Isleme  ", font=("Segoe UI", 15, "bold"),
                 fg=TEXT_MAIN, bg=BG_PANEL).pack(side='left')
        tk.Label(sol, text="Grup 12", font=("Segoe UI", 15, "bold"),
                 fg=ACCENT, bg=BG_PANEL).pack(side='left')

        # Sag: goruntu yukle + bilgi
        sag = tk.Frame(ust_bar, bg=BG_PANEL)
        sag.pack(side='right', fill='y', padx=15)

        self._goruntu_yolu_var = tk.StringVar(value="Goruntu secilmedi")
        tk.Label(sag, textvariable=self._goruntu_yolu_var,
                 font=FONT_SMALL, fg=TEXT_DIM, bg=BG_PANEL).pack(side='left', padx=10)

        HoverButton(sag, text="  Goruntu Yukle  ", font=FONT_HEAD,
                    bg=ACCENT, fg='white', relief='flat', cursor='hand2',
                    hover_bg=BTN_HOVER, padx=14, pady=4,
                    command=self._goruntu_yukle).pack(side='left', pady=10)

        # Ayirici cizgi
        tk.Frame(self, bg=BORDER, height=1).pack(fill='x')

        # ============================================================
        # SOL PANEL: goruntu onizleme + bilgi
        # ============================================================
        ana_konteyner = tk.Frame(self, bg=BG_DARK)
        ana_konteyner.pack(expand=True, fill='both')

        sol_panel = tk.Frame(ana_konteyner, bg=BG_PANEL, width=210)
        sol_panel.pack(side='left', fill='y')
        sol_panel.pack_propagate(False)

        tk.Label(sol_panel, text="Yuklulen Goruntu",
                 font=FONT_BODY, fg=TEXT_DIM, bg=BG_PANEL, pady=8).pack()

        # Onizleme kutusu
        self._onizleme_lbl = tk.Label(sol_panel, bg=BG_DARK,
                                      text="Goruntu\nyok", fg=TEXT_DIM,
                                      font=FONT_SMALL, width=26, height=12,
                                      relief='flat')
        self._onizleme_lbl.pack(padx=8, pady=(0, 8))

        # Goruntu bilgileri
        self._bilgi_var = tk.StringVar(value="—")
        tk.Label(sol_panel, textvariable=self._bilgi_var,
                 font=FONT_MONO, fg=TEXT_DIM, bg=BG_PANEL,
                 justify='left', padx=10).pack(anchor='w')

        # Cizgi
        tk.Frame(sol_panel, bg=BORDER, height=1).pack(fill='x', pady=10)

        # Kisiler listesi (durum ozeti)
        tk.Label(sol_panel, text="Grup Uyeleri",
                 font=FONT_BODY, fg=TEXT_DIM, bg=BG_PANEL, pady=4).pack()

        self._yan_menu_labels = []  # Label referanslarını tutmak için
        self._kisiler_verisi = [
            ("Kisi 1", "Temel Islemler"),
            ("Kisi 2", "Geometrik"),
            ("Kisi 3", "Filtreleme"),
            ("Kisi 4", "Kenar / Esikleme"),
            ("Kisi 5", "Morfoloji"),
        ]

        for kod, aciklama in self._kisiler_verisi:
            satir = tk.Frame(sol_panel, bg=BG_PANEL)
            satir.pack(fill='x', padx=8, pady=2)
            
            lbl_kod = tk.Label(satir, text="○ " + kod, font=FONT_BODY, fg=TEXT_DIM, bg=BG_PANEL, width=8, anchor='w')
            lbl_kod.pack(side='left')
            
            lbl_desc = tk.Label(satir, text=aciklama, font=FONT_SMALL, fg=TEXT_DIM, bg=BG_PANEL, anchor='w')
            lbl_desc.pack(side='left')
            
            self._yan_menu_labels.append((lbl_kod, lbl_desc))

        # ============================================================
        # SAG: Sekmeli panel
        # ============================================================
        sag_alan = tk.Frame(ana_konteyner, bg=BG_DARK)
        sag_alan.pack(expand=True, fill='both')

        self._notebook = ttk.Notebook(sag_alan, style='Dark.TNotebook')
        self._notebook.pack(expand=True, fill='both', padx=0, pady=0)

        # Sekmeleri olustur
        self._sekmeler = {}
        sekme_tanimlari = [
            ("Kisi 1 — Temel",     Kisi1Sekmesi),
            ("Kisi 2 — Geometrik", Kisi2Sekmesi),
            ("Kisi 3 — Filtreleme",Kisi3Sekmesi),
            ("Kisi 4 — Kenar",     Kisi4Sekmesi),
            ("● Kisi 5 — Morfoloji", Kisi5Sekmesi),
        ]

        for sekme_adi, SekmeClass in sekme_tanimlari:
            frame = SekmeClass(self._notebook, self._goruntu_geri_al,
                               style='Dark.TFrame')
            self._notebook.add(frame, text="  {}  ".format(sekme_adi))
            self._sekmeler[sekme_adi] = frame

        # Sekme degisince yan menuyu guncelle
        self._notebook.bind("<<NotebookTabChanged>>", self._sekme_degisti)

        # Kisi 5 sekmesini varsayilan sec
        self._notebook.select(4)

    def _sekme_degisti(self, event):
        """Sekme degistiginde yan menudeki [AKTIF] vurgusunu gunceller."""
        secili_index = self._notebook.index("current")
        
        for i, (lbl_kod, lbl_desc) in enumerate(self._yan_menu_labels):
            kod_metni, desc_metni = self._kisiler_verisi[i]
            
            if i == secili_index:
                # Aktif olan
                lbl_kod.config(text="● " + kod_metni, fg=SUCCESS)
                lbl_desc.config(text=desc_metni + " [AKTIF]", fg=SUCCESS)
            else:
                # Pasif olanlar
                lbl_kod.config(text="○ " + kod_metni, fg=TEXT_DIM)
                lbl_desc.config(text=desc_metni, fg=TEXT_DIM)

    # ------------------------------------------------------------------
    def _goruntu_yukle(self):
        """Dosya sec diyalog; goruntu yukle ve tum sekmelere dagit."""
        yol = filedialog.askopenfilename(
            title="Goruntu Sec",
            filetypes=[
                ("Goruntu dosyalari", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("Tum dosyalar", "*.*"),
            ]
        )
        if not yol:
            return

        # ---------------------------------------------------------------
        # TEKNİK NOT: TÜRKÇE KARAKTER VE DOSYA YOLU ÇÖZÜMÜ
        # cv2.imread doğrudan Türkçe karakterli yolları (örn: Kübra) okuyamaz.
        # Çözüm: Dosyayı önce binary (ikili) olarak belleğe alıp 
        # imdecode ile orada çözüyoruz. Böylece yol hatası yaşanmıyor.
        # ---------------------------------------------------------------
        try:
            ham_bayt = np.fromfile(yol, dtype=np.uint8)
            goruntu  = cv2.imdecode(ham_bayt, cv2.IMREAD_COLOR)
        except Exception as hata:
            messagebox.showerror("Hata", "Dosya okunamadi:\n{}".format(hata))
            return

        if goruntu is None:
            messagebox.showerror("Hata",
                "Goruntu yuklenemedi.\n"
                "Dosya bozuk veya desteklenmeyen format olabilir.\n\n"
                "Yol: {}".format(yol))
            return

        self._aktif_goruntu = goruntu
        dosya_adi = os.path.basename(yol)
        self._goruntu_yolu_var.set(dosya_adi)

        # Bilgi guncelle
        y, x = goruntu.shape[:2]
        kanallar = goruntu.shape[2] if goruntu.ndim == 3 else 1
        self._bilgi_var.set(
            "Boyut : {}x{}\nKanal : {}\nDosya : {}".format(
                x, y, kanallar, dosya_adi[:22])
        )

        # Sol panel onizleme
        photo = numpy_to_photoimage(goruntu, max_w=190, max_h=160)
        self._onizleme_lbl.config(image=photo, text='')
        self._onizleme_lbl.image = photo

        # Tum sekmelere goruntu dagit
        for sekme in self._sekmeler.values():
            try:
                sekme.goruntu_ayarla(goruntu.copy())
            except Exception:
                pass  # Placeholder sekmeler hata verirse atla

    # ------------------------------------------------------------------
    def _goruntu_geri_al(self, arr: np.ndarray):
        """Sekmenin islenmiş goruntusunu orijinal olarak ayarla (opsiyonel)."""
        pass


# ===========================================================================
# GIRIS NOKTASI
# ===========================================================================
if __name__ == "__main__":
    # Pillow kontrol
    try:
        from PIL import Image, ImageTk
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
        from PIL import Image, ImageTk

    uygulama = AnaPencere()
    uygulama.mainloop()

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
# TCL/TK YOL DUZELTME (Turk karakter iceren kullanici adi icin)
# Anaconda'nin TCL/TK kutuphane yolunu otomatik bul ve ayarla.
# Bu olmadan "Can't find a usable init.tcl" hatasi cikar.
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

# Kisi 5 modulu (bu dosya ile ayni klasorde olmali)
from kisi5_morfoloji import dilation, erosion, opening, closing
from kisi1_temel_donusumler import gri_tonlama, binary_donusum, bgr_to_hsv, histogram_germe

# from kisi2_geometrik import dondur, kirp, olcekle, aritmetik
# from kisi3_filtreleme import parlaklik_kontrast, konvolusyon, gauss, bulanik
# from kisi4_kenar     import esikleme, sobel, gurultu_ekle


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
    def __init__(self, parent, cb, **kw):
        super().__init__(parent, cb, **kw)
        self._placeholder_goster(self, 2,
            "Dondurme  |  Kirpma  |  Olcekleme  |  Aritmetik Islemler")


# ===========================================================================
# KİŞİ 3 SEKMESİ — Parlaklik, Konvolusyon, Gauss (PLACEHOLDER)
# ===========================================================================
class Kisi3Sekmesi(SekmeBaz):
    def __init__(self, parent, cb, **kw):
        super().__init__(parent, cb, **kw)
        self._placeholder_goster(self, 3,
            "Parlaklik/Kontrast  |  Konvolusyon  |  Gauss  |  Bulaniklik")


# ===========================================================================
# KİŞİ 4 SEKMESİ — Esikleme, Sobel, Gurultu (PLACEHOLDER)
# ===========================================================================
class Kisi4Sekmesi(SekmeBaz):
    def __init__(self, parent, cb, **kw):
        super().__init__(parent, cb, **kw)
        self._placeholder_goster(self, 4,
            "Esikleme (Global & Adaptif)  |  Sobel Kenar  |  Gurultu")


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

        kisiler = [
            ("Kisi 1", "Temel Islemler",    False),
            ("Kisi 2", "Geometrik",          False),
            ("Kisi 3", "Filtreleme",         False),
            ("Kisi 4", "Kenar / Esikleme",   False),
            ("Kisi 5", "Morfoloji [AKTIF]",  True),
        ]
        for kod, aciklama, aktif in kisiler:
            satir = tk.Frame(sol_panel, bg=BG_PANEL)
            satir.pack(fill='x', padx=8, pady=2)
            renk = SUCCESS if aktif else TEXT_DIM
            sembol = "●" if aktif else "○"
            tk.Label(satir, text="{} {}".format(sembol, kod),
                     font=FONT_BODY, fg=renk, bg=BG_PANEL, width=8,
                     anchor='w').pack(side='left')
            tk.Label(satir, text=aciklama, font=FONT_SMALL,
                     fg=renk if aktif else TEXT_DIM,
                     bg=BG_PANEL, anchor='w').pack(side='left')

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

        # Kisi 5 sekmesini varsayilan sec
        self._notebook.select(4)

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
        # TURKCE KARAKTER SORUNU COZUMU:
        # cv2.imread() Windows'ta Turkce/Unicode karakter iceren dosya
        # yollarini okuyamaz (ornek: "Kübra", "sağlam").
        # Cozum: dosyayi once ham bayt olarak oku (np.fromfile),
        # sonra bellekte cv2.imdecode ile coz. Yol stringi hic kullanilmaz.
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

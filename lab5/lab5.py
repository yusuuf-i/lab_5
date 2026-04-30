import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

# Eğer ortamda lena.jpg yoksa otomatik olarak indir (kodun çalışması için kolaylık)
if not os.path.exists('lena.jpg'):
    print("lena.jpg bulunamadı, indiriliyor...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    urllib.request.urlretrieve(url, 'lena.jpg')
    print("İndirme tamamlandı.")

# =======================================================
# UYGULAMA 1
# =======================================================

print("\n--- Uygulama 1 Başlıyor ---")

# 1. Görüntüyü okuma (OpenCV varsayılan olarak BGR renk uzayında okur)
img_bgr = cv.imread('lena.jpg')

if img_bgr is None:
    print("Hata: lena.jpg okunamadı! Lütfen dosyanın var olduğundan emin olun.")
    exit()

# Görüntüyü RGB, YCrCb ve HSV renk uzaylarına dönüştürme
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
img_ycrcb = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)
img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

# Sonuçları görselleştirme
# Orijinal görüntüyü ve 4 dönüşümü (RGB, BGR, YCrCb, HSV) yan yana veriyoruz
plt.figure(figsize=(15, 4))
plt.subplot(1, 5, 1); plt.imshow(img_rgb); plt.title('Orijinal')
plt.subplot(1, 5, 2); plt.imshow(img_rgb); plt.title('RGB')
plt.subplot(1, 5, 3); plt.imshow(img_bgr); plt.title('BGR')
plt.subplot(1, 5, 4); plt.imshow(img_ycrcb); plt.title('YCrCb')
plt.subplot(1, 5, 5); plt.imshow(img_hsv); plt.title('HSV')
plt.suptitle('Orijinal Görüntü ve Renk Uzayları Dönüşümleri')
plt.tight_layout()
plt.show()

# 2. Negatif Alma: g(x,y) = 255 - f(x,y)
# Sadece orijinal resme (RGB) negatif alma uyguluyoruz
img_negatif = 255 - img_rgb

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(img_rgb); plt.title('Orijinal Görüntü')
plt.subplot(1, 2, 2); plt.imshow(img_negatif); plt.title('Negatif Görüntü')
plt.suptitle('Orijinal Görüntüye Negatif Alma İşlemi (255 - f(x,y))')
plt.tight_layout()
plt.show()

# 3. İstatistiksel Analiz ve Histogram
# 3. İstatistiksel Analiz ve Histogram
# Renkli görüntü (RGB) üzerinden üç kanal için de ayrı ayrı histogram çizdirme
colors = ('r', 'g', 'b')

plt.figure(figsize=(8, 4))
plt.title('RGB Görüntü Histogramı (Kırmızı, Yeşil, Mavi)')
plt.xlabel('Piksel Değeri (0-255)')
plt.ylabel('Piksel Sayısı (Frekans)')

for i, color in enumerate(colors):
    # img_rgb üzerinde R (0), G (1), B (2) kanallarının histogramı
    hist = cv.calcHist([img_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.grid(True)
plt.show()

# İstatistiksel değerlerin raporlanması (Orijinal 3 kanallı renkli matris üzerinden)
min_val = np.min(img_rgb)
max_val = np.max(img_rgb)
median_val = np.median(img_rgb)
mean_val = np.mean(img_rgb)

print("\n--- İstatistiksel Analiz (Tüm Renkli Matris) ---")
print(f"Min Değer    : {min_val}")
print(f"Max Değer    : {max_val}")
print(f"Median Değer : {median_val}")
print(f"Mean Değer   : {mean_val:.2f}")

# Sonraki "Uygulama 2" görevleri için gri görüntü gereklidir:
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)


# =======================================================
# UYGULAMA 2: Uzamsal Filtreleme (Filtre Gezdirme)
# =======================================================

print("\n--- Uygulama 2 Başlıyor ---")

# Kenar piksellerde veri kaybını önlemek için Padding (Sıfır Ekleme) yöntemi kullanılıyor.
# OpenCV'de borderType=cv.BORDER_CONSTANT ve borderValue=0 kullanarak 0 padding yapılabilir.
pad_type = cv.BORDER_CONSTANT

# --- Gaussian ve Mean Filtresi Karşılaştırması ---
# Mean (Aritmetik Ortalama) Filtresi (3x3 ve 5x5)
mean_3x3 = cv.blur(img_rgb, (3, 3), borderType=pad_type)
mean_5x5 = cv.blur(img_rgb, (5, 5), borderType=pad_type)

# Gaussian Filtresi (3x3 ve 5x5)
gauss_3x3 = cv.GaussianBlur(img_rgb, (3, 3), 0, borderType=pad_type)
gauss_5x5 = cv.GaussianBlur(img_rgb, (5, 5), 0, borderType=pad_type)

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1); plt.imshow(img_rgb); plt.title('Orijinal')
plt.subplot(2, 3, 2); plt.imshow(mean_3x3); plt.title('Mean 3x3 (Blur)')
plt.subplot(2, 3, 3); plt.imshow(mean_5x5); plt.title('Mean 5x5 (Daha fazla Blur)')
plt.subplot(2, 3, 4); plt.imshow(img_rgb); plt.title('Orijinal')
plt.subplot(2, 3, 5); plt.imshow(gauss_3x3); plt.title('Gaussian 3x3')
plt.subplot(2, 3, 6); plt.imshow(gauss_5x5); plt.title('Gaussian 5x5')
plt.suptitle('Yumuşatma Etkileri ve Kernel Boyutu Karşılaştırması')
plt.tight_layout()
plt.show()

# --- Laplacian Filtresi (Kenar Tespiti) ---
# Kenar tespiti yüksek frekansları bulur, genelde gri görüntü üzerinde yapılır.
laplacian = cv.Laplacian(img_gray, cv.CV_64F, borderType=pad_type)
# Negatif değerleri ve float değerleri görselleştirmek için mutlak değerini alıp 8-bit'e çeviriyoruz
laplacian_abs = cv.convertScaleAbs(laplacian)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(img_gray, cmap='gray'); plt.title('Orijinal (Gri)')
plt.subplot(1, 2, 2); plt.imshow(laplacian_abs, cmap='gray'); plt.title('Laplacian (Kenarlar)')
plt.suptitle('Laplacian Filtresi ile Kenar Tespiti')
plt.show()

# --- Median Filtresi (Gürültü Temizleme) ---
# Performansı ölçmek için öncelikle yapay bir gürültü ekliyoruz (Salt & Pepper - Tuz & Biber gürültüsü)
noise_matrix = np.zeros(img_gray.shape, np.uint8)
cv.randu(noise_matrix, 0, 255) # Rastgele değerlerle doldur

noisy_img = img_gray.copy()
noisy_img[noise_matrix < 15] = 0     # %~6 Siyah noktalar (Biber)
noisy_img[noise_matrix > 240] = 255  # %~6 Beyaz noktalar (Tuz)

# Median Filtresi Uygulama
# cv.medianBlur kendi içerisinde sınır pikselleri idare eder. (3x3 veya 5x5 boyut kullanılabilir)
median_filtered_5x5 = cv.medianBlur(noisy_img, 5)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(img_gray, cmap='gray'); plt.title('Orijinal Görüntü')
plt.subplot(1, 3, 2); plt.imshow(noisy_img, cmap='gray'); plt.title('Gürültülü (Salt & Pepper)')
plt.subplot(1, 3, 3); plt.imshow(median_filtered_5x5, cmap='gray'); plt.title('Median Temizlenmiş (5x5)')
plt.suptitle('Median Filtresi ile Gürültü Temizleme Performansı')
plt.show()

# =======================================================
# UYGULAMA 3: Otsu’nun Eşikleme Metodu
# =======================================================

print("\n--- Uygulama 3 Başlıyor ---")

# Görüntü önceki adımlarda (img_gray) gri seviyeye dönüştürülmüştü.
# Otsu’s Thresholding algoritmasını uyguluyoruz.
# Threshold değeri otomatik olarak belirlenip ret_otsu değişkenine atanacak.
ret_otsu, thresh_otsu = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

print(f"Otsu Metodu ile sistemin otomatik belirlediği eşik (threshold) değeri: {ret_otsu}")

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(img_gray, cmap='gray'); plt.title('Gri Seviye Orijinal')
plt.subplot(1, 2, 2); plt.imshow(thresh_otsu, cmap='gray'); plt.title(f"Otsu İkili (Binary)\n(Threshold: {ret_otsu})")
plt.suptitle('Otsu’nun Eşikleme Metodu')
plt.show()


# =======================================================
# UYGULAMA 4: RGB ve Grayscale Arasındaki İlişki
# =======================================================

print("\n--- Uygulama 4 Başlıyor ---")

# Daha önce img_rgb olarak tanımladığımız görüntüyü R, G, B kanallarına ayırıyoruz
R, G, B = cv.split(img_rgb)

# Kanalların renklerinin daha belirgin olması için, sadece ilgili rengin dolu olduğu,
# diğer kanalların ise 0 (siyah) olduğu 3 boyutlu matrisler oluşturuyoruz.
zeros = np.zeros(img_rgb.shape[:2], dtype="uint8")

R_color = cv.merge([R, zeros, zeros])  # Matplotlib RGB sırasında olduğu için (R, 0, 0)
G_color = cv.merge([zeros, G, zeros])  # (0, G, 0)
B_color = cv.merge([zeros, zeros, B])  # (0, 0, B)

# Kanalları görselleştirme (Her bir kanal kendi renginde gösterilir ve Grayscale eklenir)
plt.figure(figsize=(18, 4))
plt.subplot(1, 5, 1); plt.imshow(img_rgb); plt.title('Orijinal RGB')
plt.subplot(1, 5, 2); plt.imshow(R_color); plt.title('Kırmızı (R) Kanalı')
plt.subplot(1, 5, 3); plt.imshow(G_color); plt.title('Yeşil (G) Kanalı')
plt.subplot(1, 5, 4); plt.imshow(B_color); plt.title('Mavi (B) Kanalı')
plt.subplot(1, 5, 5); plt.imshow(img_gray, cmap='gray'); plt.title('Grayscale Bileşimi\n(0.299R + 0.587G + 0.114B)')
plt.suptitle('RGB Görüntü, Ayrı Kanallar ve Grayscale Analizi')
plt.tight_layout()
plt.show()

# RGB -> Grayscale ilişkisini raporlama
print("\nRGB ve Grayscale (Gri Seviye) Arasındaki İlişki:")
print("OpenCV ve çoğu standart sistem, renkli görüntüyü griye çevirirken şu standart oranları (NTSC) kullanır:")
print("Gri = (0.299 * R) + (0.587 * G) + (0.114 * B)")
print("Açıklama: İnsan gözü yeşil (G) ışığa en hassas, mavi (B) ışığa ise en az hassastır.")
print("Bu yüzden en yüksek ağırlık (%58.7) Yeşil kanalına verilirken, en düşük ağırlık (%11.4) Mavi kanalına verilir.")

print("\nTüm işlemler başarıyla tamamlandı. Matplotlib pencerelerini kapattığınızda kod sonlanacaktır.")

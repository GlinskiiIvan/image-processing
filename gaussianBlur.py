import cv2
import numpy as np
import matplotlib.pyplot as plt

# Шаг 1: Загрузка изображения в градациях серого
img = cv2.imread('00005-dd5595a4.png', 0)

# Преобразование в градации серого, если изображение имеет три канала (RGB)
if img is not None and len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Шаг 2: Применение медианного фильтра для удаления шума
denoised_img = cv2.GaussianBlur(img, (5, 5), 0) # Используем ядро 5x5 и стандартное отклонение по умолчанию (0)

# Шаг 3: Отображение оригинального и сглаженного изображений
cv2.imshow('Original Image', img)
cv2.imshow('Denoised Image', denoised_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Шаг 4: Построение гистограммы сглаженного изображения
plt.figure()
plt.hist(denoised_img.ravel(), bins=256, range=(0, 256), color='black')
plt.title('Denoised Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('./gaussianBlur/denoised_histogram.png')
plt.show()

# Шаг 5: Выровняем гистограмму на сглаженном изображении
equalized_img = cv2.equalizeHist(denoised_img)

# Шаг 6: Отображение выровненного изображения
cv2.imshow('Equalized Image', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохранение выровненного изображения
cv2.imwrite('./gaussianBlur/equalized_image.png', equalized_img)

# Шаг 7: Построение гистограммы выровненного изображения
plt.figure()
plt.hist(equalized_img.ravel(), bins=256, range=(0, 256), color='black')
plt.title('Equalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('./gaussianBlur/equalized_histogram.png')
plt.show()
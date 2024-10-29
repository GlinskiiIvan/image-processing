import cv2
import numpy as np
import matplotlib.pyplot as plt

# Шаг 1: Загрузка изображения в градациях серого
img = cv2.imread('00005-dd5595a4.png', 0)

# Преобразование в градации серого, если изображение имеет три канала (RGB)
if img is not None and len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Шаг 2: Отображение оригинального изображения
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Шаг 3: Построение и сохранение гистограммы
plt.figure()
plt.hist(img.ravel(), bins=256, range=(0, 256), color='black')
plt.title('Original Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('./noise/original_histogram.png')  # Сохранение гистограммы
plt.show()

# Шаг 4: Выровняем гистограмму
equalized_img = cv2.equalizeHist(img)

# Отображение выровненного изображения
cv2.imshow('Equalized Image', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохранение выровненного изображения
cv2.imwrite('./noise/equalized_image.png', equalized_img)

# Шаг 5: Построение и сохранение гистограммы выровненного изображения
plt.figure()
plt.hist(equalized_img.ravel(), bins=256, range=(0, 256), color='black')
plt.title('Equalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.savefig('./noise/equalized_histogram.png')  # Сохранение гистограммы выровненного изображения
plt.show()
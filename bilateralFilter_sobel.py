import cv2
import numpy as np
import matplotlib.pyplot as plt

# Шаг 1: Загрузка изображения в градациях серого
img = cv2.imread('00005-dd5595a4.png', 0)

# Шаг 2: Применение билатерального фильтра для удаления шума
denoised_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Шаг 3: Применение CLAHE для выравнивания гистограммы
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_img = clahe.apply(denoised_img)

# Шаг 4: Усиление краев с использованием оператора Собеля
sobel_x = cv2.Sobel(equalized_img, cv2.CV_64F, 1, 0, ksize=3)  # Условие по x
sobel_y = cv2.Sobel(equalized_img, cv2.CV_64F, 0, 1, ksize=3)  # Условие по y
edges = cv2.magnitude(sobel_x, sobel_y)

# Преобразование обратно в 8-битное изображение
edges = cv2.convertScaleAbs(edges)

# Шаг 5: Отображение оригинального и сглаженного изображений
cv2.imshow('Original Image', img)
cv2.imshow('Denoised Image (Bilateral Filter)', denoised_img)
cv2.imshow('Equalized Image (CLAHE)', equalized_img)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Шаг 6: Сохранение изображений
cv2.imwrite('./bilateralFilter_sobel/denoised_image.png', denoised_img)
cv2.imwrite('./bilateralFilter_sobel/equalized_image_clahe.png', equalized_img)
cv2.imwrite('./bilateralFilter_sobel/edges_image.png', edges)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Шаг 1: Загрузка изображения в градациях серого
img = cv2.imread('00005-dd5595a4.png', 0)

# Шаг 2: Применение билатерального фильтра для удаления шума
denoised_img = cv2.medianBlur(img, 3)  # Используем ядро 3x3 для сглаживания

# Шаг 3: Применение CLAHE для выравнивания гистограммы
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_img = clahe.apply(denoised_img)

# Шаг 4: Усиление краев с использованием оператора Лапласа
laplacian = cv2.Laplacian(equalized_img, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)  # Преобразование обратно в 8-битное изображение

# Шаг 5: Отображение оригинального и обработанных изображений
cv2.imshow('Original Image', img)
cv2.imshow('Denoised Image (Bilateral Filter)', denoised_img)
cv2.imshow('Equalized Image (CLAHE)', equalized_img)
cv2.imshow('Laplacian Edges', laplacian_abs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Шаг 6: Сохранение изображений
cv2.imwrite('./medianBlur_laplacian/denoised_image.png', denoised_img)
cv2.imwrite('./medianBlur_laplacian/equalized_image_clahe.png', equalized_img)
cv2.imwrite('./medianBlur_laplacian/laplacian_edges_image.png', laplacian_abs)

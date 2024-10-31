import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Указание пути для сохранения
save_path = './clahe_medianBlur/'
# Создание директории, если она не существует
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Шаг 1: Загрузка изображения в градациях серого
img = cv2.imread('00005-dd5595a4.png', 0)

# Шаг 2: Применение медианного фильтра для удаления шума
denoised_img = cv2.medianBlur(img, 3)  # Используем ядро 3x3 для сглаживания

# Шаг 3: Применение CLAHE для выравнивания гистограммы
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_img = clahe.apply(denoised_img)

# Шаг 4: Отображение оригинального и сглаженного изображений
cv2.imshow('Original Image', img)
cv2.imshow('Denoised Image', denoised_img)

# Шаг 5: Отображение выровненного изображения
cv2.imshow('Equalized Image', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохранение выровненного изображения
cv2.imwrite(save_path + '/denoised_image.png', denoised_img)
cv2.imwrite(save_path + '/equalized_image.png', equalized_img)
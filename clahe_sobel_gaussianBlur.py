import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Указание пути для сохранения
save_path = './clahe_sobel_gaussianBlur/'
# Создание директории, если она не существует
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Шаг 1: Загрузка изображения в градациях серого
img = cv2.imread('00005-dd5595a4.png', 0)

# Шаг 2: Применение фильтра гауса для удаления шума
denoised_img = cv2.GaussianBlur(img, (5, 5), 0) # Используем ядро 5x5 и стандартное отклонение по умолчанию (0)

# Шаг 3: Применение CLAHE для выравнивания гистограммы
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_img = clahe.apply(denoised_img)

# Шаг 4: Усиление краев с использованием оператора Собеля
sobel_x = cv2.Sobel(equalized_img, cv2.CV_64F, 1, 0, ksize=3)  # Условие по x
sobel_y = cv2.Sobel(equalized_img, cv2.CV_64F, 0, 1, ksize=3)  # Условие по y
edges = cv2.magnitude(sobel_x, sobel_y)

# Преобразование обратно в 8-битное изображение
edges = cv2.convertScaleAbs(edges)

# Совмещаем изображение с его границами для усиления контуров
edges_image = cv2.addWeighted(equalized_img, 1.0, edges, 0.5, 0)

# Шаг 5: Отображение оригинального и сглаженного изображений
cv2.imshow('Original Image', img)
cv2.imshow('Denoised Image (Bilateral Filter)', denoised_img)
cv2.imshow('Equalized Image (CLAHE)', equalized_img)
cv2.imshow('Edges', edges_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Шаг 6: Сохранение изображений
cv2.imwrite(save_path + '/denoised_image.png', denoised_img)
cv2.imwrite(save_path + '/equalized_image.png', equalized_img)
cv2.imwrite(save_path + '/edges_image.png', edges_image)
